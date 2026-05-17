"""AI Atlas Nexus policy generation for Mellea Skills Compiler."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

from mellea_skills_compiler.enums import GovernanceTaxonomy, InferenceEngineType
from mellea_skills_compiler.inference import InferenceService
from mellea_skills_compiler.models import GovernanceAction, NexusRisk, PolicyManifest
from mellea_skills_compiler.toolkit.logging import configure_logger


LOGGER = configure_logger()


# Native Granite Guardian risks that should ALWAYS be checked, regardless of
# what Nexus identifies for the specific use case. These target the hallucination
# / off-topic failure modes that surface as quality bugs in compiled skills
# (the 2026-05-15 empirical audit showed Nexus's per-use-case risk set —
# typically harm + social_bias + jailbreak — caught zero of the hallucination
# findings on the eu-ai-act-classification fixture).
#
# - `groundedness`: checks whether the model output is grounded in the
#   provided context. Catches fabricated facts the model wasn't given input
#   for (e.g. "Energy-Intensive Digital Services Provider" sector inferred
#   when no sector was in the scenario).
# - `answer_relevance`: checks whether the response addresses the actual
#   query. Catches off-topic drift.
#
# Both are native dimensions in Granite Guardian (4.x and later); the
# `guardian_prompt` is the bare tag the model recognises.
MANDATORY_GUARDIAN_RISKS: list[NexusRisk] = [
    NexusRisk(
        name="Groundedness",
        description=(
            "Verify the response is grounded in the provided context. "
            "Flags claims the model could not have derived from the input "
            "(hallucinations, fabricated entities, invented citations)."
        ),
        guardian_prompt="groundedness",
        is_native=True,
        taxonomy=GovernanceTaxonomy.IBM_GRANITE_GUARDIAN,
    ),
    NexusRisk(
        name="Answer Relevance",
        description=(
            "Verify the response addresses the actual user query rather than "
            "drifting to adjacent topics. Flags off-topic responses and "
            "non-answers."
        ),
        guardian_prompt="answer_relevance",
        is_native=True,
        taxonomy=GovernanceTaxonomy.IBM_GRANITE_GUARDIAN,
    ),
]


def generate_policy_manifest(
    use_case: str,
    nexus,
    model: Optional[str] = None,
    inference_engine: InferenceEngineType = InferenceEngineType.OLLAMA,
    governance_taxonomies: Optional[list[str]] = None,
) -> PolicyManifest:
    """Identify applicable risks and governance actions to produce a multi-taxonomy policy manifest.

    Args:
        use_case (str): Natural language description of the agent's purpose.
        nexus (AIAtlasNexus): AI Atlas Nexus instance.
        model (Optional[str], optional): Model to use for Risk and Action Identification. The `inference_engine` param must support the model. If set to None, the default model for the inference engine will be used.
        inference_engine (InferenceEngineType, optional): Service to use for LLM inference. Defaults to InferenceEngineType.OLLAMA.
        governance_taxonomies: List of taxonomy IDs for governance actions.
    Returns:
        PolicyManifest with Guardian risks and governance actions.
    """

    if not governance_taxonomies:
        governance_taxonomies = GovernanceTaxonomy.list()

    # ── 1. Runtime risk checks ────────────
    LOGGER.info("Identifying risks for use case: %.100s...", use_case)

    # Create Inference engine instance
    risk_inference_engine = InferenceService(inference_engine).risk(
        model, parameters={"temperature": 0}
    )

    risk_lists = nexus.identify_risks_and_actions_from_usecases(
        [use_case],
        risk_inference_engine,
        taxonomy=governance_taxonomies,
        zero_shot_only=True,
    )

    identified_risks = risk_lists.get("risks", [])
    LOGGER.info("AI Atlas Nexus: %d risks identified", len(identified_risks))

    nexus_risks = []
    nexus_additional_risks = []
    all_governance_risk_names = []

    for risk in identified_risks:
        desc = risk.description or ""

        guardian_prompt = desc.strip()
        is_native = False

        all_governance_risk_names.append(risk.name)

        # sort the risks, granite guardian and other
        if risk.isDefinedByTaxonomy == GovernanceTaxonomy.IBM_GRANITE_GUARDIAN:
            if risk.tag:
                guardian_prompt = risk.tag
                is_native = True

            nexus_risks.append(
                NexusRisk(
                    name=risk.name,
                    description=desc,
                    guardian_prompt=guardian_prompt,
                    is_native=is_native,
                    taxonomy=risk.isDefinedByTaxonomy,
                )
            )
            tier = "native" if is_native else "custom"
            LOGGER.info(
                "  [Guardian] %s (%s) → %s", risk.name, tier, guardian_prompt[:60]
            )
        else:
            nexus_additional_risks.append(
                NexusRisk(
                    name=risk.name,
                    description=desc,
                    guardian_prompt=guardian_prompt,
                    is_native=is_native,
                    taxonomy=risk.isDefinedByTaxonomy,
                )
            )

    # Inject MANDATORY_GUARDIAN_RISKS (groundedness, answer_relevance) that
    # should always be checked, regardless of Nexus's per-use-case output.
    # Deduplicated by guardian_prompt so Nexus-identified versions of the
    # same risk are not overwritten.
    existing_prompts = {r.guardian_prompt for r in nexus_risks}
    for mandatory in MANDATORY_GUARDIAN_RISKS:
        if mandatory.guardian_prompt in existing_prompts:
            continue
        nexus_risks.append(mandatory)
        all_governance_risk_names.append(mandatory.name)
        LOGGER.info(
            "  [Guardian] %s (native, mandatory) → %s",
            mandatory.name,
            mandatory.guardian_prompt,
        )

    # -- 2. Use the actions which are directly linked the risks
    identified_risks_governance_actions = risk_lists.get("mixed_control_items", [])
    all_governance_actions: list[GovernanceAction] = []
    for governance_item in identified_risks_governance_actions:
        all_governance_actions.append(
            GovernanceAction(
                id=governance_item.id,
                name=governance_item.name or governance_item.id,
                description=governance_item.description or "",
                source=governance_item.isDefinedByTaxonomy,
                category="",
                via_risk="",
                categorized_as=governance_item.isCategorizedAs,
            )
        )

    return PolicyManifest(
        use_case=use_case,
        taxonomy=governance_taxonomies,
        risks=nexus_risks,
        additional_risks=nexus_additional_risks,
        governance_actions=all_governance_actions,
        governance_taxonomies_used=governance_taxonomies,
        governance_risks_identified=all_governance_risk_names,
        model_used=risk_inference_engine.model_name_or_path,
    )


def generate_policy_markdown(manifest: PolicyManifest) -> str:
    """Generate a human-readable .md policy document from a manifest.

    Includes:
      - Guardian runtime risk checks (what the hooks monitor)
      - NIST AI RMF governance actions (organisational requirements)
      - Credo UCF controls (specific mitigation measures)
      - Guardrail configuration table
      - Audit trail specification
    """
    all_taxonomies = manifest.governance_taxonomies_used
    lines = [
        f"# Policy: {manifest.use_case}",
        "",
        f"**Generated**: {manifest.generated_at}  ",
        f"**Risk identification model**: {manifest.model_used}  ",
        f"**Taxonomies**: {', '.join(all_taxonomies)}",
    ]
    lines.extend(["", "---", ""])

    # ── Section 1: Guardian runtime checks ──────────────────────────
    lines.extend(
        [
            "## 1. Runtime Risk Checks (Granite Guardian)",
            "",
            "The following risks are checked at runtime on every LLM generation "
            "via the configured Granite Guardian model (see "
            "`mellea_skills_compiler.enums:OLLAMA_GUARDIAN_MODEL`). Risk "
            "descriptions are sourced from the IBM Granite Guardian taxonomy "
            "in AI Atlas Nexus, with `groundedness` and `answer_relevance` "
            "always included as mandatory native risks for hallucination / "
            "off-topic detection. Each native risk's `guardian_prompt` is the "
            "bare tag (e.g. `harm`, `groundedness`); custom risks send the "
            "full description as Guardian criteria.",
            "",
        ]
    )

    for i, risk in enumerate(manifest.risks, 1):
        tier_label = "Native dimension" if risk.is_native else "Custom criteria"
        lines.extend(
            [
                f"### 1.{i} {risk.name}",
                "",
                f"**Guardian tier**: {tier_label}  ",
                f"**Guardian prompt**: `{risk.guardian_prompt}`",
                "",
                f"{risk.description}",
                "",
            ]
        )

    # ── Section 2: Governance actions (per taxonomy) ────────────────
    if manifest.governance_actions:
        # Group actions by source taxonomy
        by_source: dict[str, list[GovernanceAction]] = defaultdict(list)
        for action in manifest.governance_actions:
            by_source[action.source].append(action)

        section_num = 2
        for source, actions in by_source.items():
            lines.extend(
                [
                    "---",
                    "",
                    f"## {section_num}. Governance Requirements ({source})",
                    "",
                    f"**{len(actions)}** governance actions collected from "
                    f"the **{source}** taxonomy.",
                    "",
                ]
            )

            # Sub-group by category if categories exist (e.g. NIST Govern/Map/Measure/Manage)
            by_cat: dict[str, list[GovernanceAction]] = defaultdict(list)
            for action in actions:
                by_cat[action.category or "General"].append(action)

            cat_order = ["Govern", "Map", "Measure", "Manage", "General", "Other"]
            for cat in cat_order:
                cat_actions = by_cat.get(cat, [])
                if not cat_actions:
                    continue
                if cat != "General" or len(by_cat) > 1:
                    lines.extend([f"### {section_num}.{cat[0]}. {cat}", ""])
                for action in cat_actions:
                    desc = action.description
                    if len(desc) > 300:
                        desc = desc[:297] + "..."
                    lines.extend(
                        [
                            f"- **[{action.id}]** {desc}",
                            "",
                        ]
                    )

            section_num += 1

    # ── Guardrail configuration ──────────────────────────────────────
    # section_num is already set from the governance loop (or 2 if no governance)
    if not manifest.governance_actions:
        section_num = 2
    lines.extend(
        [
            "---",
            "",
            f"## {section_num}. Guardrail Configuration",
            "",
            "| Risk | Tier | Guardian Prompt | Hook | Mode | Priority |",
            "|------|------|----------------|------|------|----------|",
        ]
    )
    for risk in manifest.risks:
        tier = "Native" if risk.is_native else "Custom"
        lines.append(
            f"| {risk.name} | {tier} | `{risk.guardian_prompt}` "
            f"| `generation_post_call` | AUDIT | 40 |"
        )

    # ── Audit trail ──────────────────────────────────────────────────
    section_num += 1
    lines.extend(
        [
            "",
            f"## {section_num}. Audit Trail",
            "",
            "All generation events are logged to `audit_trail.jsonl` with:",
            "- Guardian verdicts per risk (from Section 1)",
            "- Component lifecycle events (pre-execute, post-success, post-error)",
            "- Validation outcomes",
            "- Policy manifest ID for traceability back to this document",
            "",
            "The governance guidance sections are organisational requirements — "
            "they inform how the agent should be deployed, monitored, and "
            "governed, complementing the automated runtime checks in Section 1.",
        ]
    )

    return "\n".join(lines)


def load_policy_manifest(audit_dir: Path) -> PolicyManifest:
    """Search for policy_manifest.json in standard locations.

    Checks the skill root first (portable), then the audit directory.
    """
    manifest_path = audit_dir / "policy_manifest.json"
    if manifest_path.is_file():
        try:
            return PolicyManifest.from_json(str(manifest_path))
        except Exception as e:
            raise Exception(
                f"Failed to load policy manifest from {manifest_path}: {str(e)}",
            )

    raise Exception(f"No policy_manifest.json found in {audit_dir}")
