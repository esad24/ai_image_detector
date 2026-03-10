

## Setup

### 1. Clone the repository

```bash
git clone 
cd ai_image_detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```


### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

---

## Running the Detection Pipeline

```bash
python src/main.py -i  -m  -p  [-r]
```

### Arguments

| Flag | Description |
|------|-------------|
| `-i` | Image folder path or dataset shortcut (see below) |
| `-m` | Model name |
| `-p` | Prompt ID (integer) |
| `-r` | Resume from a previous run (skip already processed images) |

### Dataset Shortcuts (`-i`)

| Shortcut | Path |
|----------|------|
| `genClass_fake` | `data/genClass/fake/images` |
| `genClass_real` | `data/genClass/real/images` |
| `genArtifact_fake` | `data/genArtifact/fake/test/images` |
| `genArtifact_real` | `data/genArtifact/real/test/images` |
| `real2gen_fake` | `data/real2gen/fake/images` |
| `real2gen_real` | `data/real2gen/real/images` |

### Supported Models (`-m`)

| Model name | Backend |
|------------|---------|
| `gpt-5.2` | OpenAI API |
| `claude-sonnet-4.5` | Anthropic API |
| `qwen3-vl` | Ollama (local) |
| `kimi-2.5` | Ollama (local) |

### Examples

```bash
# Run GPT-5.2 on fake test images with prompt 3
python src/main.py -i genArtifact_fake -m gpt-5.2 -p 3

# Resume an interrupted run
python src/main.py -i genArtifact_fake -m gpt-5.2 -p 3 -r

# Use a custom folder
python src/main.py -i /path/to/my/images -m claude-sonnet-4.5 -p 2
```

Results are saved to:
```
data/<split>/test/results/<model>/prompt<id>/<timestamp>/results.json
```

---


### Evaluation Code needs to be adjusted 

