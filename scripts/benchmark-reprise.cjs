// Run after pnpm run build. Arguments: llama.cpp model ID, output JSON.
const fs = require("node:fs");
const { execFileSync } = require("node:child_process");
const { Chat } = require("@lmstudio/sdk");
const implementation = process.env.BENCH_COMPACTOR || "../compaction.js";
const { summarize, buildChat, foldRepeatedLines, renderTranscript } = require(
  implementation,
);

const scenarios = [
  {
    name: "Livraisons",
    turns: [
      "Le 3 mai et le 8 mai, deux livraisons de fraises sont arrivées abîmées. Les deux lots ont été refusés.",
      "Le 12 mai, une troisième livraison de fraises est arrivée abîmée. Cette fois, le lot a été accepté avec une remise de 15 %.",
      "Le fournisseur propose un remplacement pour le 16 mai. La date attend encore la confirmation du magasin.",
    ],
    question:
      "Quels incidents ont eu lieu, quelle décision a été prise pour chaque lot et où en est le remplacement ?",
    receipt: "Le personnel a signé la feuille de présence.",
  },
  {
    name: "Windows",
    turns: [
      "Je veux déplacer les projets et le dossier .codex sur D:.",
      "La jonction des projets est en place. Les hachages et les liens physiques des projets ont été vérifiés.",
      "La tentative de déplacer .codex a échoué sur les métadonnées de sécurité. La nouvelle tentative s'est arrêtée car Codex était ouvert. Aucune réussite confirmée.",
    ],
    question: "Quelle migration a été vérifiée et laquelle reste à confirmer ?",
  },
  {
    name: "Naruto",
    turns: [
      "Je veux vérifier le correctif des compétences et le combat Kyubi.",
      "Le correctif des données de compétences est installé sur le téléphone.",
      "Le combat Kyubi a été testé uniquement sur BlueStacks. Les phases tardives restent à vérifier.",
    ],
    question:
      "Sur quel appareil le combat Kyubi a-t-il été testé, et quel changement a été installé sur le téléphone ?",
  },
  {
    name: "FMSeries",
    turns: [
      "Comparons l'accord de risque et l'accord exact pendant le rejeu.",
      "Sur cette unique session synthétique, l'accord de risque a anticipé 10 épisodes WARNING sur 10, contre 9 sur 10 pour l'accord exact.",
      "Les fenêtres d'évaluation se chevauchent. On compte 8 fenêtres de fausse alerte pour le risque et 7 pour l'accord exact. La généralisation reste à établir.",
    ],
    question:
      "Que montrent les résultats pour les deux accords et quelles limites empêchent de généraliser ?",
  },
];

async function main() {
  const [id, output] = process.argv.slice(2);
  if (!id || !output) throw new Error("Expected model ID and output JSON path");
  const base = process.env.BENCH_URL || "http://127.0.0.1:18431";
  async function request(path, body) {
    const response = await fetch(
      base + path,
      body && {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    );
    if (!response.ok)
      throw new Error(
        path + " " + response.status + " " + (await response.text()),
      );
    return response.json();
  }
  const props = await request("/props");
  const context = props.default_generation_settings.n_ctx;
  const messagesOf = (chat) =>
    chat
      .getMessagesArray()
      .map((m) => ({ role: m.getRole(), content: m.getText() }));
  const model = {
    getContextLength: async () => context,
    countTokens: async (content) =>
      (await request("/tokenize", { content, parse_special: true })).tokens
        .length,
    applyPromptTemplate: async (chat) =>
      (await request("/apply-template", { messages: messagesOf(chat) })).prompt,
    respond: async (chat, options) =>
      (
        await request("/v1/chat/completions", {
          model: id,
          messages: messagesOf(chat),
          max_tokens: options.maxTokens,
          temperature: 0,
          seed: 42,
        })
      ).choices[0].message,
  };
  const count = async (chat) =>
    model.countTokens(await model.applyPromptTemplate(chat));
  const answer = async (chat) =>
    (await model.respond(chat, { temperature: 0, maxTokens: 180 })).content;
  const report = {
    revision: execFileSync("git", ["rev-parse", "HEAD"], {
      encoding: "utf8",
    }).trim(),
    model: id,
    modelFile: require("node:path").basename(props.model_path || id),
    backend: "llama.cpp",
    implementationSha256: require("node:crypto")
      .createHash("sha256")
      .update(fs.readFileSync(require.resolve(implementation)))
      .digest("hex"),
    context,
    scenarios: [],
  };
  const suite =
    process.argv[4] === "long"
      ? require("./long-conversations.json")
      : scenarios;
  for (const scenario of suite) {
    if (process.argv[5] && scenario.name !== process.argv[5]) continue;
    const history = Chat.empty();
    const cache = new Map();
    const checkpoints = [];
    report.scenarios.push({ ...scenario, checkpoints });
    const turns = scenario.dialogue || scenario.turns.map((turn) => [turn]);
    for (const [index, [turn, reply]] of turns.entries()) {
      history.append("user", turn);
      if (reply !== undefined) {
        history.append("assistant", reply);
        if (index !== turns.length - 1) continue;
      } else {
        // Neutral interleaved tool-like output, identical across cases.
        const log = Array.from(
          { length: 12 },
          (_, i) =>
            "Contrôle " +
            (checkpoints.length * 12 + i + 1) +
            " | lecture du journal temporaire | lot traité | aucun diagnostic supplémentaire.",
        ).join("\n");
        history.append(
          "assistant",
          scenario.receipt
            ? Array(12).fill(scenario.receipt).join("\n")
            : "Consigné. Journal intermédiaire :\n" + log,
        );
      }
      const prefix = Array.from(history.getMessagesArray());
      const stages = [];
      let stage;
      const tracedModel = {
        ...model,
        async respond(chat, options) {
          const response = await model.respond(chat, options);
          stages.push({
            stage,
            input: chat.at(0).getText(),
            output: response.content,
          });
          return response;
        },
      };
      const result = await summarize(tracedModel, prefix, {
        signal: new AbortController().signal,
        cache,
        onProgress: (label) => {
          stage = label;
        },
        chunkTokens: 512,
        rawContextTokens: context,
        countTokens: (message) => model.countTokens(message.getText()),
      });
      result.stages = stages;
      checkpoints.push(result);
      if (scenario.dialogue) {
        const sourceLines = scenario.dialogue.flat();
        result.distinctMessages = new Set(sourceLines).size;
        result.messageCount = sourceLines.length;
        if (foldRepeatedLines) {
          const transcript = renderTranscript(prefix);
          result.foldingChangedText =
            foldRepeatedLines(transcript) !== transcript;
        }
      }
    }
    history.append("user", scenario.question);
    const messages = history.getMessagesArray();
    const final = checkpoints.at(-1);
    const compact = buildChat(history, messages, 0, {
      splitIndex: messages.length - 1,
      summary: final.summary,
    });
    for (const [label, chat] of [
      ["full", history],
      ["compacted", compact],
    ]) {
      final[label] = { tokens: await count(chat), answer: await answer(chat) };
    }
    fs.writeFileSync(output, JSON.stringify(report, null, 2) + "\n");
    console.log(scenario.name, final.full.tokens, final.compacted.tokens);
  }
}
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
