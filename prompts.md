# Prompt Definitions

Six prompt variants used to instruct LLM detectors. They range from simple binary classification to detailed artifact reasoning, including two adversarial control prompts (prompt5 & prompt6).

---

## Prompt 1 — Binary Classification

**Strategy:** Simple yes/no detection. The model is asked to classify the image as real or fake with no further explanation.

**Prompt text:**
> Tell me if this is an AI-generated image. Answer real or fake. Respond ONLY with valid JSON

**Output schema:**
| Field | Type | Values |
|-------|------|--------|
| `classification` | string (enum) | `"real"` \| `"fake"` |

---

## Prompt 2 — Artifact Type Detection

**Strategy:** If the image is fake, the model must also identify which *categories* of artifacts are present — without any explanation or location.

**Prompt text:**
> Tell me if this is an AI-generated image. If it is fake identify which artifact types are present using only the following categories: 'structural' (Structural & Component Defects), 'physics' (violation of the laws of physics), 'semantic' (Semantic & Functional Implausibilities), 'stylistic'. Do not provide any explanation. Respond ONLY with valid JSON

**Output schema:**
| Field | Type | Values |
|-------|------|--------|
| `classification` | string (enum) | `"real"` \| `"fake"` |
| `artifacts` | array of objects | — |
| `artifacts[].type` | string (enum) | `"structural"` \| `"physical"` \| `"semantic"` \| `"stylistic"` |

---

## Prompt 3 — Detailed Artifact Reasoning

**Strategy:** Most detailed prompt. For each artifact the model must provide the type, a textual description of the anomaly, and its spatial location in the image. Results go through a manual review step.

**Prompt text:**
> Tell me if this is an AI-generated image. If the image is fake, identify all visible artifacts using only the following categories: 'structural' (Structural & Component Defects), 'physics' (violation of the laws of physics), 'semantic' (Semantic & Functional Implausibilities), 'stylistic'. For each artifact, specify its type, describe the anomaly and indicate its spatial location in the image. Respond ONLY with valid JSON

**Output schema:**
| Field | Type | Values |
|-------|------|--------|
| `classification` | string (enum) | `"real"` \| `"fake"` |
| `artifacts` | array of objects | — |
| `artifacts[].type` | string (enum) | `"structural"` \| `"physical"` \| `"semantic"` \| `"stylistic"` |
| `artifacts[].reasoning` | string | Free-text description of the anomaly |
| `artifacts[].location` | string | Spatial location within the image |

---

## Prompt 4 — Open-ended Artifact Explanation

**Strategy:** Asks the model to detect synthesis artifacts without constraining artifact types to a fixed taxonomy. The model must explain *why* and *where* artifacts exist.

**Prompt text:**
> Tell me if synthesis artifacts are in this image. Must return with 1) fake or real; 2) if real, explain why and where the artifacts exist. Respond ONLY with valid JSON

**Output schema:**
| Field | Type | Values |
|-------|------|--------|
| `classification` | string (enum) | `"real"` \| `"fake"` |
| `artifacts` | array of objects | — |
| `artifacts[].reasoning` | string | Free-text explanation of the artifact |
| `artifacts[].location` | string | Spatial location within the image |

---

## Prompt 5 — Adversarial Control (Forced Fake)

**Strategy:** Control / adversarial prompt. The model is *told* the image is fake and asked to justify it. Used to measure whether models produce plausible-sounding but fabricated explanations when given a false premise. Applied only on **real images**.

**Prompt text:**
> This is a fake image. Explain why and where the artifacts exist. Respond ONLY with valid JSON

**Output schema:**
| Field | Type | Values |
|-------|------|--------|
| `classification` | string (enum) | `"real"` \| `"fake"` |
| `artifacts` | array of objects | — |
| `artifacts[].reasoning` | string | Free-text explanation |
| `artifacts[].location` | string | Spatial location within the image |

---

## Prompt 6 — Adversarial Control (Forced Real)

**Strategy:** Mirror of prompt5. The model is *told* the image is real and asked to justify it. Applied only on **fake images** to test whether models comply with a false premise.

**Prompt text:**
> This is a real image. Explain why. Respond ONLY with valid JSON

**Output schema:**
| Field | Type | Values |
|-------|------|--------|
| `classification` | string (enum) | `"real"` \| `"fake"` |
| `artifacts` | array of objects | — |
| `artifacts[].reasoning` | string | Free-text explanation |
| `artifacts[].location` | string | Spatial location within the image |

---

## Summary

| Prompt | Strategy | Artifact Types | Reasoning | Location | Applied On |
|--------|----------|:--------------:|:---------:|:--------:|------------|
| prompt1 | Binary classification | — | — | — | Fake & Real |
| prompt2 | Artifact type detection | ✓ | — | — | Fake & Real |
| prompt3 | Detailed artifact reasoning | ✓ | ✓ | ✓ | Fake & Real |
| prompt4 | Open-ended explanation | — | ✓ | ✓ | Fake & Real |
| prompt5 | Adversarial — forced fake | — | ✓ | ✓ | Real only |
| prompt6 | Adversarial — forced real | — | ✓ | ✓ | Fake only |