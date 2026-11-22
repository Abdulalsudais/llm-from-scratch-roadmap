import React, { useState, useEffect, useMemo, useRef } from "react";

// =========================================================================
// 1. ROADMAP CURRICULUM, PIPELINE & ENVIRONMENT SETUP DATA
// =========================================================================
const PIPELINE_STAGES = [
  {
    id: 1,
    name: "1. Tokenizer",
    sub: "Text → Token IDs",
    phaseId: "phase-3",
    badge: "Phase 3 • Tokenization",
    desc: 'Raw text is split into subword integer IDs using Byte-Pair Encoding: "The capital of France is" → [464, 3139, 286, 4881, 318].'
  },
  {
    id: 2,
    name: "2. Embeddings",
    sub: "IDs → Dense Vectors",
    phaseId: "phase-3",
    badge: "Phase 3 • Vector Space",
    desc: "Each token ID looks up a learned vector (e.g., 768 floats) plus positional info (RoPE/Positional Embedding) so order matters."
  },
  {
    id: 3,
    name: "3. Self-Attention",
    sub: "Context Mixing (Q·Kᵀ)",
    phaseId: "phase-4",
    badge: "Phase 4 • Heart of GPT",
    desc: "Tokens communicate via Queries, Keys, and Values with a causal mask so 'France' attends back to 'capital' without peeking at future words."
  },
  {
    id: 4,
    name: "4. Feed-Forward",
    sub: "MLP Knowledge Store",
    phaseId: "phase-4",
    badge: "Phase 4 • Transformer Block",
    desc: "Each token passes through LayerNorm/RMSNorm, a 2-layer MLP with GELU/SwiGLU activation, and residual skip connections."
  },
  {
    id: 5,
    name: "5. LM Head + Softmax",
    sub: "Logits → Probabilities",
    phaseId: "phase-5",
    badge: "Phase 5 • Next-Token Prediction",
    desc: "Final hidden vectors project back to vocabulary size (e.g., 50,257 logits). Softmax turns them into probabilities—'Paris' scores highest!"
  },
  {
    id: 6,
    name: "6. SFT & Local Run",
    sub: "Assistant Alignment",
    phaseId: "phase-6",
    badge: "Phase 6 • Post-Training",
    desc: "Fine-tune the base predictor on Q&A instructions (SFT/LoRA), quantize weights, and serve locally via Ollama or FastAPI."
  }
];

const SETUP_SOFTWARES = [
  {
    name: "Python 3.11 (64-bit)",
    role: "Core runtime for scripts, training loops & PyTorch",
    whereUsed: "Used by OpenAI, Meta, and Hugging Face as the primary orchestration language for training pipelines, tokenizers, and inference APIs.",
    whatYouBuild: "Your custom BPE tokenizer, micrograd autograd engine, GPT model classes, and training loop.",
    url: "https://www.python.org/downloads/",
    winget: "winget install -e --id Python.Python.3.11"
  },
  {
    name: "Visual Studio Code",
    role: "Code editor, integrated terminal & Python debugger",
    whereUsed: "Used across AI engineering teams to inspect multi-dimensional PyTorch tensor shapes, debug training crashes, and run Jupyter notebooks side-by-side.",
    whatYouBuild: "Your modular Python project files (tokenizer.py, model.py, train.py, app.py) with live variable inspection.",
    url: "https://code.visualstudio.com/",
    winget: "winget install -e --id Microsoft.VisualStudioCode"
  },
  {
    name: "Git for Windows + GitHub",
    role: "Version control, commits, branches & repo hosting",
    whereUsed: "Used by every open-source and enterprise AI team (such as nanoGPT and Llama repos) to track model architecture experiments and roll back broken hyperparameter changes.",
    whatYouBuild: "Versioned checkpoints of your LLM codebase as you progress from micrograd to full GPT pretraining.",
    url: "https://git-scm.com/downloads/win",
    winget: "winget install -e --id Git.Git"
  },
  {
    name: "Ollama",
    role: "Run small pretrained models locally on your PC",
    whereUsed: "Used by developers and enterprises for private, offline local LLM inference (running quantized Llama 3, Qwen, or Gemma models on consumer laptops).",
    whatYouBuild: "A local reference model server to benchmark latency, prompt templates, and output quality against your from-scratch GPT.",
    url: "https://ollama.com/download",
    winget: "winget install -e --id Ollama.Ollama"
  }
];

const SETUP_PACKAGES = [
  {
    name: "torch",
    desc: "PyTorch deep learning & tensor autograd engine",
    whereUsed: "Powers Meta Llama 3, Tesla AI, and most frontier research labs for GPU matrix math and automatic backpropagation.",
    whatYouBuild: "Self-Attention heads, LayerNorm, SwiGLU MLPs, LoRA adapters, and your full Decoder-Only GPT."
  },
  {
    name: "numpy",
    desc: "Fast C-powered multi-dimensional array math",
    whereUsed: "Used across every ML data pipeline for fast CPU array manipulation, vector similarity, and preprocessing before GPU transfer.",
    whatYouBuild: "Vector cosine-similarity visualizers, Softmax/Cross-Entropy prototypes, and memory-mapped token arrays."
  },
  {
    name: "transformers",
    desc: "Reference model architectures & pretrained weights",
    whereUsed: "Industry standard library for downloading open-source checkpoints (GPT-2, Llama, Gemma) and verifying architecture parity.",
    whatYouBuild: "Weight-loading scripts that copy open GPT-2 124M weights directly into your custom PyTorch classes."
  },
  {
    name: "tokenizers",
    desc: "Fast Rust-backed BPE subword tokenizers",
    whereUsed: "Used in production LLM servers to encode/decode 100k+ token vocabularies in microseconds.",
    whatYouBuild: "High-speed reference tokenizers to benchmark against your pure-Python BPE tokenizer."
  },
  {
    name: "datasets",
    desc: "Load, stream & slice large text corpora cleanly",
    whereUsed: "Used to stream terabytes of pretraining text (FineWeb, TinyStories, WikiText) without running out of laptop RAM.",
    whatYouBuild: "Clean train/validation text pipelines feeding sliding-window token chunks into your GPT."
  },
  {
    name: "jupyterlab",
    desc: "Interactive notebooks for step-by-step testing",
    whereUsed: "Used by AI researchers to inspect attention heatmaps, tensor shapes, and loss curves cell-by-cell.",
    whatYouBuild: "Interactive scratchpads for micrograd graphs, causal mask matrices, and prompt generation tests."
  },
  {
    name: "pandas",
    desc: "Tabular dataset inspection & cleaning",
    whereUsed: "Used in SFT/fine-tuning workflows to filter noisy Q&A pairs, check prompt lengths, and deduplicate records.",
    whatYouBuild: "Clean instruction-tuning datasets formatted for Supervised Fine-Tuning (SFT)."
  },
  {
    name: "matplotlib",
    desc: "Plot training/validation loss & LR curves",
    whereUsed: "Used to spot overfitting, exploding gradients, and learning-rate warmup schedules visually.",
    whatYouBuild: "Train vs. validation loss charts, cosine LR curves, and attention-weight heatmaps."
  },
  {
    name: "pytest",
    desc: "Automated unit tests for tensor shapes & outputs",
    whereUsed: "Used in production ML CI/CD pipelines to ensure architectural edits don't break output shapes or determinism.",
    whatYouBuild: "Automated shape tests and 20-prompt regression checks for your fine-tuned assistant."
  },
  {
    name: "mlflow",
    desc: "Track experiment metrics, hyperparams & runs",
    whereUsed: "Used by ML teams to compare perplexity, learning rates, and checkpoints across dozens of training runs.",
    whatYouBuild: "A local experiment dashboard logging your GPT's loss and perplexity at every step."
  },
  {
    name: "fastapi",
    desc: "Wrap your LLM in a typed local REST API",
    whereUsed: "Powers high-throughput Python AI microservices and OpenAI-compatible inference gateways.",
    whatYouBuild: "A POST /v1/generate web endpoint with Pydantic validation serving your custom GPT."
  },
  {
    name: "uvicorn",
    desc: "Fast local ASGI web server to run FastAPI",
    whereUsed: "Used in production containers to serve concurrent HTTP requests to FastAPI model endpoints.",
    whatYouBuild: "The live localhost server hosting your LLM API and interactive /docs Swagger UI."
  },
  {
    name: "requests",
    desc: "Send HTTP test prompts to local or cloud APIs",
    whereUsed: "Used in client scripts and evaluation harnesses to query LLM endpoints and measure response latency.",
    whatYouBuild: "A Python client script that sends prompts to both your FastAPI GPT and Ollama to compare outputs."
  },
  {
    name: "lxml",
    desc: "Fast XML/HTML corpus parsing helper",
    whereUsed: "Used when cleaning raw web scrapes, Wikipedia dumps, or structured documentation into plain training text.",
    whatYouBuild: "Clean corpus extractors stripping HTML/XML tags before BPE tokenization."
  }
];

const SETUP_COMMANDS_TEXT = `# 1. Create project folder & isolated Python 3.11 virtual environment
mkdir LLM-From-Scratch
cd LLM-From-Scratch
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1

# 2. Upgrade pip & install all LLM learning, tracking & API packages
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib jupyterlab transformers datasets tokenizers pytest lxml requests mlflow fastapi uvicorn

# 3. Install PyTorch (CPU default — for NVIDIA CUDA GPU see https://pytorch.org/get-started/locally/)
python -m pip install torch

# 4. Verify everything works
python --version
git --version
ollama --version
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA GPU:', torch.cuda.is_available())"
python -c "import transformers, datasets, tokenizers, mlflow, fastapi, uvicorn, jupyterlab; print('All LLM packages OK!')"`;

const ROADMAP_PHASES = [
  {
    id: "phase-1",
    number: "01",
    category: "foundations",
    accent: "teal",
    duration: "Week 1–2",
    title: "Python, NumPy & PyTorch Foundations for LLMs",
    subtitle: "Master clean Python OOP, NumPy broadcasting, the (B, T, C) tensor shape mental model, and probability/loss math before writing neural nets.",
    plainBigPicture: "An LLM is really just a giant calculator that reads numbers representing words and predicts the probability of the next word. Before building the brain, you need Python (the glue language), NumPy (a fast math library for grids of numbers), and PyTorch (an AI framework that runs those grids on your GPU).",
    liveAiBridge: {
      headline: "Where Live Running AIs Use Phase 1 Skills",
      summary: "Every time you type a prompt into ChatGPT, Claude, or Llama 3, the server packages your conversation into a 3D PyTorch tensor (Batch, Tokens, Hidden_Dim), runs matrix multiplications on NVIDIA GPUs, and uses Softmax to pick the next word's probability.",
      examples: [
        "OpenAI & Anthropic APIs: Your 'temperature' and 'logprobs' sliders directly manipulate the Softmax logits math you learn in Module 1.5.",
        "Meta Llama 3 & Mistral configs: Open-source model repos define their entire architecture using the exact Python @dataclass config pattern from Module 1.1.",
        "vLLM / PyTorch Inference Servers: Production GPUs batch multiple users' chats together along the 'B' (Batch) axis you master in Module 1.3."
      ]
    },
    jargonDecoder: [
      { term: "LLM", def: "Next-word probability predictor" },
      { term: "NumPy", def: "Fast grid-of-numbers math library" },
      { term: "PyTorch", def: "AI framework with GPU acceleration" },
      { term: "Tensor", def: "Multi-dimensional spreadsheet of numbers" },
      { term: "Logits", def: "Raw unscaled word scores" },
      { term: "Softmax", def: "Converts raw scores into % probabilities" }
    ],
    achievements: [
      "Confidently write clean, modular Python classes and @dataclass configs inside an isolated .venv environment.",
      "Trace (Batch, Sequence, Embedding) 3D tensor shapes across matrix multiplications without shape-mismatch errors.",
      "Convert raw model logits into calibrated token probabilities using Softmax and compute Cross-Entropy loss."
    ],
    miniProject: "Build a Matrix Multiplication & Softmax Temperature Visualizer in NumPy/PyTorch",
    topics: [
      {
        id: "p1-t1",
        codeNum: "1.1",
        title: "Python Data Structures, OOP & Clean ML Code",
        duration: "2 Days",
        plainEnglish: "Python is the programming language we use to orchestrate everything. Because an AI model has hundreds of configurable knobs (layers, heads, vocabulary size), we bundle them cleanly inside Python 'Classes' (reusable blueprints) and isolate our tools in a '.venv' folder using 'pip' so nothing breaks your computer.",
        liveExample: "Meta's official Llama-3 repository and Hugging Face Transformers define every model's blueprint (ModelArgs dataclass + nn.Module classes) using this exact Python OOP pattern—allowing engineers to switch between an 8B and 70B model simply by changing config fields.",
        jargonTerms: [
          { term: "Python 3.11", def: "The human-readable programming language used to write AI scripts." },
          { term: ".venv (Virtual Env)", def: "An isolated sandbox folder holding project packages so different projects never clash." },
          { term: "pip", def: "Python's built-in package installer (like an app store) used to download libraries like PyTorch." },
          { term: "OOP (Classes)", def: "Grouping data and functions together into reusable blueprints (like a 'Car' blueprint)." },
          { term: "@dataclass", def: "A clean Python shortcut for creating settings/config containers without boilerplate." }
        ],
        why: "Every PyTorch model is built by subclassing classes, overriding forward(), and managing token dictionaries and configuration dataclasses cleanly.",
        concepts: [
          "Isolated Python 3.11 virtual environments (.venv), pip dependency locking, and JupyterLab cell debugging.",
          "Python classes, inheritance (super().__init__()), and dunder methods (__call__, __len__, __getitem__, __repr__).",
          "Clean configuration management with @dataclass, type hints, and generator expressions."
        ],
        code: `from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257    # Total unique word-pieces our model knows
    context_length: int = 256  # Max words the model can read at once
    emb_dim: int = 384         # How many numbers describe each word
    n_heads: int = 6           # Number of parallel attention spotlights
    n_layers: int = 6          # How many reasoning blocks are stacked

cfg = GPTConfig()
print(f"Model width: {cfg.emb_dim}, Head dim: {cfg.emb_dim // cfg.n_heads}")`,
        pitfall: "Never install ML packages globally into your OS Python—always verify 'which python' or 'Get-Command python' points inside your project's .venv folder.",
        project: "Write a clean TokenizerBase Python class with __len__, encode(), and decode() methods plus a @dataclass config object.",
        resourceTitle: "Sebastian Raschka — Build an LLM From Scratch (Environment & Python Setup)",
        resourceNote: "Start here to structure clean Python classes and virtual environments before writing PyTorch models.",
        links: [
          { type: "YouTube", label: "Sebastian Raschka — Build an LLM from Scratch 1: Set Up Your Code Environment", url: "https://www.youtube.com/watch?v=yAcWnfsZhzo" },
          { type: "YouTube", label: "Andrej Karpathy — Neural Networks: Zero to Hero (Full Playlist)", url: "https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ" },
          { type: "Community Guide", label: "Real Python — Object-Oriented Programming & Data Classes Guide", url: "https://realpython.com/python-data-classes/" },
          { type: "Blog / Docs", label: "Python Official Docs — Data Classes & OOP Reference", url: "https://docs.python.org/3/library/dataclasses.html" }
        ]
      },
      {
        id: "p1-t2",
        codeNum: "1.2",
        title: "Vectors, Matrices & Broadcasting in NumPy",
        duration: "3 Days",
        plainEnglish: "Standard Python lists are far too slow for AI because Python checks data types one item at a time. NumPy stores numbers in contiguous C-memory blocks so your CPU can multiply millions of numbers simultaneously. A 'Vector' is just a 1D list of numbers, and a 'Matrix' is a 2D table of numbers.",
        liveExample: "Semantic search engines, Pinecone/Chroma vector databases, and ChatGPT's RAG file-upload search turn your PDF paragraphs into vectors and run the exact dot-product cosine similarity (X @ X.T) from this module to find the most relevant page in milliseconds.",
        jargonTerms: [
          { term: "NumPy", def: "Numerical Python—a C-powered library for ultra-fast array math." },
          { term: "Vector", def: "A 1D list of numbers (e.g., [0.2, -0.5, 0.8]) representing a word's meaning." },
          { term: "Matrix", def: "A 2D grid (rows × columns) of numbers, like an Excel sheet." },
          { term: "Dot Product (@)", def: "Multiplying matching numbers and summing them to measure similarity." },
          { term: "Broadcasting", def: "Automatically stretching a smaller row/column to add onto a bigger table." }
        ],
        why: "Token embeddings are vectors, weight layers are matrices, and NumPy broadcasting rules govern how biases and attention masks combine automatically.",
        concepts: [
          "Scalars (0D), Vectors (1D), Matrices (2D), and Tensors (3D/4D) as nested numeric arrays.",
          "Element-wise operations vs. dot products (np.dot / @) as geometric cosine similarity.",
          "Broadcasting rules: aligning trailing dimensions (e.g., adding a (1, C) bias vector to a (T, C) token matrix)."
        ],
        code: `import numpy as np

X = np.array([[1.0, 0.2, -0.5],
              [0.9, 0.1, -0.4],
              [-0.8, 0.7, 0.3],
              [0.0, -0.5, 0.9]])

bias = np.array([0.1, -0.1, 0.05])
print("Broadcasted sum shape:", (X + bias).shape)
print("Pairwise similarity (4x4):\\n", np.round(X @ X.T, 2))`,
        pitfall: "Confusing element-wise multiplication (*) with matrix multiplication (@). In Python, A * B multiplies matching cells; A @ B performs row-by-column dot products.",
        project: "Implement vector cosine similarity in NumPy and test how similar word vectors have dot products near +1 while unrelated vectors sit near 0.",
        resourceTitle: "3Blue1Brown — Essence of Linear Algebra (Chapters 1 & 4)",
        resourceNote: "Visualizes vectors, linear transformations, matrix multiplication, and dot products geometrically.",
        links: [
          { type: "YouTube", label: "3Blue1Brown — Vectors | Essence of Linear Algebra Ch. 1", url: "https://www.youtube.com/watch?v=fNk_zzaMoSs" },
          { type: "YouTube", label: "3Blue1Brown — Matrix Multiplication as Composition Ch. 4", url: "https://www.youtube.com/watch?v=XkY2DOUCWMU" },
          { type: "Community Guide", label: "Jay Alammar — A Visual Intro to NumPy and Data Representation", url: "https://jalammar.github.io/visual-numpy/" },
          { type: "Blog / Docs", label: "NumPy Official Manual — Array Broadcasting Visual Guide", url: "https://numpy.org/doc/stable/user/basics.broadcasting.html" }
        ]
      },
      {
        id: "p1-t3",
        codeNum: "1.3",
        title: "Matrix Multiplication & Tensor Shapes (B, T, C) Mental Model",
        duration: "3 Days",
        plainEnglish: "Think of a 'Tensor' as a stack of spreadsheets. Throughout an entire LLM, data travels in a 3D box shaped (B, T, C): B is how many sentences we process at once (Batch), T is how many words are in each sentence (Time/Tokens), and C is how many numbers describe each word (Channels).",
        liveExample: "When ChatGPT serves thousands of users simultaneously, its inference engine groups 64 users' prompts into one (B=64, T=512, C=4096) tensor so a single GPU matrix multiplication computes the next word for all 64 users at the exact same time.",
        jargonTerms: [
          { term: "Tensor", def: "A generalized container of numbers (1D list, 2D table, or 3D+ cube)." },
          { term: "B (Batch Size)", def: "Number of separate sentences processed in parallel for speed." },
          { term: "T (Sequence Length)", def: "Number of tokens (word-pieces) inside a single sentence." },
          { term: "C (Embedding Dim)", def: "Size of the number list describing each individual token." }
        ],
        why: "Every single Transformer layer passes around a 3D tensor of shape (Batch, Time/Sequence, Channels/Embedding). Knowing this mental model prevents 90% of bugs.",
        concepts: [
          "What B (batch size), T (sequence length / context window), and C (embedding dimension) represent physically.",
          "Batched matrix multiplication: (B, T, C) @ (C, D) → (B, T, D) transforms every token independently in parallel.",
          "Reshaping vs. transposing: why .view()/.reshape() and .transpose(-2, -1) behave differently in memory."
        ],
        code: `import torch

B, T, C = 2, 5, 64   # 2 sentences (B), 5 words each (T), 64 numbers per word (C)
W_out = 128

x = torch.randn(B, T, C)   # (2, 5, 64)
W = torch.randn(C, W_out)  # (64, 128)
y = x @ W                  # (2, 5, 128)
print("Transformed tensor shape:", y.shape)`,
        pitfall: "Never use .view() to swap axes like (B, T, C) to (B, C, T)—that scrambles data! Always use .transpose(1, 2) when swapping dimensions.",
        project: "Write a shape-tracing script that passes a dummy (4, 16, 128) tensor through projections, splits it into 4 heads (4, 4, 16, 32), and merges it back.",
        resourceTitle: "Jay Alammar — The Illustrated Transformer (Tensor Dimension Diagrams)",
        resourceNote: "Study how input matrices keep their sequence length T constant from bottom to top of the stack.",
        links: [
          { type: "Community Guide", label: "Jay Alammar — The Illustrated Transformer (Must-Read Visual Blog)", url: "https://jalammar.github.io/illustrated-transformer/" },
          { type: "YouTube", label: "3Blue1Brown — Transformers, the tech behind LLMs (Ch. 5)", url: "https://www.youtube.com/watch?v=wjZofJX0v4M" },
          { type: "Community Guide", label: "GeeksforGeeks — Tensor Dimensions & Matrix Multiplication in Deep Learning", url: "https://www.geeksforgeeks.org/matrix-multiplication-in-neural-networks/" }
        ]
      },
      {
        id: "p1-t4",
        codeNum: "1.4",
        title: "PyTorch Tensors, Device Management & Vectorization",
        duration: "2 Days",
        plainEnglish: "PyTorch is like NumPy with two superpowers: it can run calculations on a graphics card (GPU) 50× faster than a CPU, and it automatically remembers every math operation so it can teach the AI later. 'Vectorization' simply means doing math on the whole grid at once instead of slow Python loops.",
        liveExample: "Every open-weight model you run locally via Ollama, LM Studio, or Hugging Face uses PyTorch's .to(device) and vectorized tensor kernels to automatically route math onto your NVIDIA CUDA GPU or Apple Silicon M-series chip without rewriting a single math equation.",
        jargonTerms: [
          { term: "PyTorch", def: "Meta's open-source Python library for building and training neural networks." },
          { term: "CUDA / GPU", def: "Graphics card hardware designed to do thousands of matrix multiplications at once." },
          { term: "MPS", def: "Apple Silicon (M1/M2/M3/M4) GPU acceleration backend for PyTorch on Mac." },
          { term: "Vectorization", def: "Replacing slow one-by-one Python loops with single-step array math." },
          { term: "dtype (float32)", def: "How much memory each number uses (e.g., 32-bit decimal vs 64-bit integer)." }
        ],
        why: "PyTorch replaces slow Python for-loops with vectorized C++/CUDA tensor kernels and lets the exact same code run on CPU, NVIDIA CUDA, or Apple MPS.",
        concepts: [
          "Creating tensors, inspecting .shape, .dtype (float32, bfloat16, int64), and .device.",
          "Device-agnostic code pattern: device = 'cuda' if torch.cuda.is_available() else 'cpu'.",
          "Vectorization: replacing nested Python loops with fast tensor reductions (.sum(dim=-1), .mean(dim=-1, keepdim=True))."
        ],
        code: `import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(32, 128, device=device, dtype=torch.float32)

mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
print(f"Running on {device} | Mean shape: {mean.shape}")`,
        pitfall: "Forgetting keepdim=True during reductions! Without keepdim=True, (32, 128).mean(dim=-1) becomes (32,) instead of (32, 1), causing silent wrong broadcasting.",
        project: "Benchmark a pure-Python nested loop dot-product against vectorized PyTorch tensor operations on 1,000×1,000 matrices.",
        resourceTitle: "PyTorch Official Tutorials & Sebastian Raschka GitHub Appendix A",
        resourceNote: "Covers tensor creation, data types, GPU device transfers, and essential tensor ops.",
        links: [
          { type: "Community Guide", label: "MachineLearningMastery — PyTorch Tensors & GPU Operations Guide", url: "https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models-with-python/" },
          { type: "YouTube", label: "PyTorch Official — Introduction to PyTorch Tensors", url: "https://www.youtube.com/watch?v=r7QDUPb2dCM" },
          { type: "Blog / Docs", label: "PyTorch Official Tutorial — Introduction to Tensors", url: "https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html" }
        ]
      },
      {
        id: "p1-t5",
        codeNum: "1.5",
        title: "Probability Distributions, Logits, Softmax & Cross-Entropy Loss",
        duration: "3 Days",
        plainEnglish: "When an LLM guesses the next word, it outputs a raw score ('Logit') for every word in its dictionary—like +2.5 for 'Paris' and -1.0 for 'banana'. 'Softmax' squashes those raw scores into percentages adding up to 100%, and 'Cross-Entropy Loss' is the penalty score grading how wrong the guess was.",
        liveExample: "In the OpenAI Playground and API, enabling 'logprobs=True' exposes the exact Softmax token percentages you compute here—showing why ChatGPT was 94.2% sure the next word was 'Paris' vs. 3.1% for 'Lyon'.",
        jargonTerms: [
          { term: "Logits", def: "Raw positive or negative scores before converting to percentages." },
          { term: "Softmax", def: "Math formula turning any list of numbers into probabilities between 0% and 100%." },
          { term: "Cross-Entropy Loss", def: "Penalty score: 0 means 100% confident & right; high means confident & wrong." },
          { term: "Vocabulary", def: "The complete dictionary of all tokens the model knows (e.g., 50,257)." }
        ],
        why: "An LLM outputs unnormalized raw scores (logits) across 50,000+ vocabulary words. Softmax turns them into probabilities, and Cross-Entropy grades how confident the model was in the true next token.",
        concepts: [
          "Logits: raw positive or negative scores emitted by the final linear layer before normalization.",
          "Numerically stable Softmax: subtracting max(logits) before exp() so large numbers never overflow to NaN.",
          "Negative Log-Likelihood / Cross-Entropy Loss: -log(P(true_token))—why random guessing over 50,257 tokens starts at loss ≈ ln(50257) ≈ 10.82."
        ],
        code: `import torch, torch.nn.functional as F

logits = torch.tensor([[2.5, 0.5, -1.0]]) # Raw scores for 3 possible next words
target = torch.tensor([0])                 # Index 0 is the actual right word

probs = F.softmax(logits, dim=-1)          # Convert to percentages summing to 1.0
loss = F.cross_entropy(logits, target)     # Penalty score (lower is better)
print(f"Probs: {probs.tolist()} | Cross-Entropy Loss: {loss.item():.4f}")`,
        pitfall: "Passing Softmax probabilities into F.cross_entropy()! PyTorch's F.cross_entropy expects raw unnormalized logits because it applies log_softmax internally.",
        project: "Build a Softmax & Cross-Entropy visualizer notebook showing how initial random LLM loss equals ln(vocab_size) and drops toward 0 as target probability approaches 1.",
        resourceTitle: "StatQuest — Cross-Entropy & Softmax Clearly Explained",
        resourceNote: "Intuitive walkthrough of log-probabilities and why Cross-Entropy penalizes confident wrong predictions heavily.",
        links: [
          { type: "YouTube", label: "StatQuest — Neural Networks Part 6: Cross Entropy", url: "https://www.youtube.com/watch?v=6ArSys5qHAU" },
          { type: "Community Guide", label: "Victor Zhou — Softmax & Cross-Entropy Loss Explained Simply", url: "https://victorzhou.com/blog/softmax/" },
          { type: "Community Guide", label: "Dive into Deep Learning — Softmax Regression from Scratch", url: "https://d2l.smola.org/chapter_linear-classification/softmax-regression-scratch.html" }
        ]
      }
    ]
  },
  {
    id: "phase-2",
    number: "02",
    category: "math-pytorch",
    accent: "indigo",
    duration: "Week 3–4",
    title: "Neural Networks & Autograd from Scratch",
    subtitle: "Demystify 'magic' training loops by building a scalar backpropagation engine (micrograd) before graduating to PyTorch nn.Module.",
    plainBigPicture: "How does an AI actually 'learn' from its mistakes? Every calculation is recorded as a step-by-step recipe (a Computation Graph). When the model makes a wrong guess, 'Backpropagation' walks backward through that recipe using basic calculus to figure out which knobs ('Weights') to nudge slightly up or down.",
    liveAiBridge: {
      headline: "Where Live Running AIs Use Phase 2 Skills",
      summary: "Every frontier model (GPT-4, Gemini, Llama 3, DeepSeek) acquired its coding, reasoning, and translation skills by running the exact PyTorch autograd + AdamW + SwiGLU loop you build in Phase 2 across billions of text passages.",
      examples: [
        "Llama 3, Mistral & Gemma MLP Layers: Two-thirds of the weights inside these live models are the exact SwiGLU feed-forward networks you code in Module 2.3.",
        "Frontier Pretraining Clusters: OpenAI, Meta, and xAI train multi-billion-parameter models using the exact AdamW optimizer + gradient accumulation loop from Module 2.4.",
        "Production Inference Mode: Every live chatbot runs behind model.eval() (Module 2.5) so Dropout layers stop dropping neurons when answering real users."
      ]
    },
    jargonDecoder: [
      { term: "Weights", def: "Adjustable numbers inside the AI brain" },
      { term: "Gradient", def: "Direction to nudge a weight to reduce error" },
      { term: "Backpropagation", def: "Walking backward to assign blame for errors" },
      { term: "Autograd", def: "PyTorch's automatic gradient calculator" },
      { term: "AdamW", def: "Smart algorithm that updates the weights" }
    ],
    achievements: [
      "Explain exactly what happens inside loss.backward() because you built a working scalar Autograd engine (micrograd) yourself.",
      "Construct multi-layer neural networks with modern non-linear activations (GELU & SwiGLU) and proper weight initialization.",
      "Write bug-free PyTorch training loops using AdamW, zero_grad(), train()/eval() switching, and activation diagnostics."
    ],
    miniProject: "Code Micrograd from Scratch & Train a 2-Layer Neural Network Without PyTorch",
    topics: [
      {
        id: "p2-t1",
        codeNum: "2.1",
        title: "Computation Graphs & The Chain Rule",
        duration: "2 Days",
        plainEnglish: "Imagine a chain of dominoes or gears: turning Gear A turns Gear B twice as fast, and Gear B turns Gear C three times as fast—so Gear C moves 2 × 3 = 6 times as fast as Gear A. That simple multiplication of step-by-step sensitivities is the 'Chain Rule', and it's all you need to train an LLM.",
        liveExample: "During fine-tuning of a customer-support LLM, PyTorch builds a dynamic computation graph linking the final wrong word back through 32 Transformer layers—using the Chain Rule to pinpoint which attention weights caused the factual error.",
        jargonTerms: [
          { term: "Computation Graph", def: "A flowchart showing every add/multiply step from input to final error." },
          { term: "Derivative / Gradient", def: "How much the final error changes if you nudge one number slightly." },
          { term: "Chain Rule", def: "Multiplying step-by-step sensitivities along a chain of operations." }
        ],
        why: "Every neural network—from a 2-line equation to GPT-4—is a Directed Acyclic Graph (DAG) of elementary additions and multiplications differentiated via the Chain Rule.",
        concepts: [
          "Derivatives as local sensitivity: measuring how nudging an input by h changes the final loss.",
          "Decomposing complex expressions (e.g., L = (w*x + b - y)²) into intermediate nodes.",
          "The Chain Rule: multiplying upstream gradient (dL/dout) by local derivative (dout/din)."
        ],
        code: `x, y = 2.0, 10.0
w, b = 3.0, 1.0
z = w * x + b              # Prediction: 7.0
diff = z - y               # Error gap: -3.0
loss = diff ** 2           # Squared error: 9.0

dloss_ddiff = 2 * diff     # -6.0
dloss_dw = dloss_ddiff * x # -12.0 (increase w to lower loss!)`,
        pitfall: "Overwriting gradients (=) instead of accumulating them (+=) when a variable is used in multiple branches of a graph!",
        project: "Draw a 5-node computation graph on paper and verify your hand-calculated gradients against numerical finite differences: (f(x+h) - f(x)) / h.",
        resourceTitle: "3Blue1Brown — Backpropagation, Intuitively (Deep Learning Ch. 3)",
        resourceNote: "Builds rock-solid geometric intuition for how gradients flow backward through expressions.",
        links: [
          { type: "YouTube", label: "3Blue1Brown — Backpropagation, Intuitively (Deep Learning Ch. 3)", url: "https://www.youtube.com/watch?v=Ilg3gGewQ5U" },
          { type: "Community Guide", label: "Chris Olah's Blog — Calculus on Computational Graphs: Backpropagation", url: "https://colah.github.io/posts/2015-08-Backprop/" },
          { type: "YouTube", label: "Andrej Karpathy — Becoming a Backprop Ninja (makemore Part 4)", url: "https://www.youtube.com/watch?v=q8SA3rM6ckI" }
        ]
      },
      {
        id: "p2-t2",
        codeNum: "2.2",
        title: "Building micrograd: Scalar Autograd Engine",
        duration: "4 Days",
        plainEnglish: "When you call 'loss.backward()' in PyTorch, it feels like magic. Here you will build your own tiny 100-line clone of PyTorch's brain ('micrograd') using plain Python numbers so you see every line of code behind automatic differentiation.",
        liveExample: "PyTorch's C++ autograd engine and Google JAX's autodiff system are scaled-up tensor versions of your micrograd Value class—recording every forward math op on a tape and replaying ._backward() in reverse topological order during training.",
        jargonTerms: [
          { term: "Autograd", def: "Automatic differentiation—computing gradients automatically without hand-math." },
          { term: "Backpropagation", def: "Running the Chain Rule backward from the final loss to every weight." },
          { term: "Topological Sort", def: "Ordering graph steps so a node only runs backward after all nodes after it finish." }
        ],
        why: "Writing Karpathy's ~100-line micrograd engine removes all mystery from PyTorch's autograd—you will understand exactly what loss.backward() executes.",
        concepts: [
          "Wrapping floats in a Value class storing .data, .grad, ._prev children, and a ._backward() closure.",
          "Implementing __add__, __mul__, __pow__, tanh(), and relu() with local gradient rules.",
          "Topological sorting of the computation DAG so backward() visits every node in strict reverse dependency order."
        ],
        code: `class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data, self.grad = float(data), 0.0
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out`,
        pitfall: "Forgetting to reset node.grad = 0.0 before each training iteration—accumulated gradients from previous steps will corrupt parameter updates.",
        project: "Implement the complete Value autograd engine in Python and verify its gradients against PyTorch's torch.autograd on a multi-operation expression.",
        resourceTitle: "Andrej Karpathy — Building micrograd (Zero to Hero Ep. 1)",
        resourceNote: "Code along line-by-line in JupyterLab; this is the single highest-ROI exercise for LLM beginners.",
        links: [
          { type: "YouTube", label: "Andrej Karpathy — The spelled-out intro to neural networks: building micrograd", url: "https://www.youtube.com/watch?v=VMj-3S1tku0" },
          { type: "Community Guide", label: "Andrej Karpathy — Official micrograd GitHub Repository", url: "https://github.com/karpathy/micrograd" },
          { type: "Community Guide", label: "Towards Data Science — Building an Autograd Engine From Scratch", url: "https://karpathy.ai/zero-to-hero.html" }
        ]
      },
      {
        id: "p2-t3",
        codeNum: "2.3",
        title: "Neurons, Layers, MLPs & Non-Linear Activations (ReLU, GELU, SwiGLU)",
        duration: "3 Days",
        plainEnglish: "A single artificial 'Neuron' simply multiplies inputs by weights, adds them up, and passes the result through a curve ('Activation Function' like GELU or SwiGLU) so the network can learn complex patterns instead of only straight lines. Stacking layers of neurons creates an 'MLP'—the memory bank inside every Transformer block.",
        liveExample: "AI interpretability researchers (such as Anthropic's mechanistic interpretability team) discovered that factual knowledge like 'The Eiffel Tower is in Paris' is stored inside the exact MLP neuron layers you build here—and Llama 3 & Qwen use your exact SwiGLU block.",
        jargonTerms: [
          { term: "Neuron", def: "A tiny math unit: weighted sum of inputs + bias + activation curve." },
          { term: "MLP", def: "Multi-Layer Perceptron—a stack of linear layers with activation curves between them." },
          { term: "Activation Function", def: "A non-linear bend (ReLU, GELU, SwiGLU) allowing AI to learn complex rules." },
          { term: "nn.Linear", def: "PyTorch's built-in matrix multiplication layer (y = x @ W.T + b)." }
        ],
        why: "Two-thirds of the parameters in a Transformer block live inside its Multi-Layer Perceptron (MLP), where non-linear activations unlock complex reasoning.",
        concepts: [
          "Single Neuron → Layer of Neurons → Multi-Layer Perceptron (MLP) architecture.",
          "Why non-linearities are essential: stacking linear layers without activations collapses mathematically into a single matrix multiply.",
          "Modern LLM activations: ReLU vs. smooth GELU (GPT-2/3) vs. gated SwiGLU (Llama 3 / Gemma)."
        ],
        code: `import torch, torch.nn as nn, torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up   = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))`,
        pitfall: "Dead ReLU neurons: if weights initialize poorly or learning rate is too high, negative pre-activations output 0 forever with zero gradient—one reason LLMs prefer GELU/SwiGLU.",
        project: "Build Neuron, Layer, and MLP classes on top of your micrograd Value class and train them to solve a non-linear 2D classification dataset.",
        resourceTitle: "StatQuest & 3Blue1Brown — Neural Networks & Activations",
        resourceNote: "Visualizes how layered neurons compose edges and patterns into higher-level representations.",
        links: [
          { type: "YouTube", label: "StatQuest — The Essential Main Ideas of Neural Networks", url: "https://www.youtube.com/watch?v=CqOfi41LfDw" },
          { type: "Community Guide", label: "Papers With Code — SwiGLU & GELU Activation Functions Explained", url: "https://paperswithcode.com/method/swiglu" },
          { type: "Community Guide", label: "GeeksforGeeks — Activation Functions in Neural Networks (ReLU, GELU, SiLU)", url: "https://www.geeksforgeeks.org/activation-functions-neural-networks/" }
        ]
      },
      {
        id: "p2-t4",
        codeNum: "2.4",
        title: "Gradient Descent, Learning Rate & AdamW Optimizer",
        duration: "3 Days",
        plainEnglish: "Imagine standing blindfolded on a mountain and trying to reach the lowest valley by feeling the slope under your feet. 'Gradient Descent' takes steps downhill; 'Learning Rate' is your step size; and 'AdamW' is a smart GPS optimizer that takes bigger steps on flat terrain and smaller, careful steps on steep cliffs.",
        liveExample: "Virtually every major open and proprietary LLM—including GPT-3, Llama 3, Qwen, and Mistral—is trained with AdamW using betas=(0.9, 0.95) and weight_decay=0.1, the exact hyperparameters in your code snippet.",
        jargonTerms: [
          { term: "Gradient Descent", def: "Stepping opposite to the gradient to steadily shrink prediction error." },
          { term: "Learning Rate (lr)", def: "How big of a nudge to apply to weights on each step (e.g., 0.0003)." },
          { term: "Optimizer (AdamW)", def: "Smart algorithm that adjusts step sizes individually for every single weight." },
          { term: "Weight Decay", def: "Gently pulling weights toward zero so the model doesn't memorize noise." }
        ],
        why: "Plain Stochastic Gradient Descent (SGD) struggles on Transformers because embedding weights and attention projections have wildly different gradient scales. AdamW adapts step sizes per parameter.",
        concepts: [
          "Vanilla Gradient Descent update rule: param = param - lr * param.grad.",
          "Momentum (exponential moving average of past gradients) and RMSProp (adaptive scaling by gradient magnitude).",
          "AdamW: combining adaptive per-parameter learning rates with decoupled weight decay regularization."
        ],
        code: `import torch

model = torch.nn.Linear(64, 64)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

# Canonical 3-line PyTorch parameter update:
# optimizer.zero_grad(set_to_none=True)
# loss.backward()
# optimizer.step()`,
        pitfall: "Using a huge learning rate like lr=0.1 (common in basic SGD demos) with AdamW on a Transformer—LLM AdamW learning rates typically sit around 1e-4 to 6e-4.",
        project: "Compare SGD, SGD+Momentum, and AdamW on a noisy 2D Rosenbrock bowl in Python and plot their optimization trajectories.",
        resourceTitle: "StatQuest — Gradient Descent & Adam Optimizer Intuition",
        resourceNote: "Clear visual explanation of why momentum and adaptive learning rates speed up convergence.",
        links: [
          { type: "YouTube", label: "StatQuest — Gradient Descent, Step-by-Step", url: "https://www.youtube.com/watch?v=sDv4f4s2SB8" },
          { type: "Community Guide", label: "Distill.pub — Why Momentum Really Works (Interactive Visual Essay)", url: "https://distill.pub/2017/momentum/" },
          { type: "Community Guide", label: "Sebastian Ruder — An Overview of Gradient Descent Optimization Algorithms", url: "https://www.ruder.io/optimizing-gradient-descent/" }
        ]
      },
      {
        id: "p2-t5",
        codeNum: "2.5",
        title: "Training Loop Anatomy, Initialization & Vanishing/Exploding Gradients",
        duration: "3 Days",
        plainEnglish: "Before training starts, all weights are filled with tiny random numbers ('Initialization'). If those random numbers are too big, calculations blow up to infinity ('Exploding Gradients'); if too tiny, signals fade to zero ('Vanishing Gradients'). A clean 'Training Loop' repeats guess → grade → nudge thousands of times safely.",
        liveExample: "Whenever a live API serves an LLM in production, the deployment pipeline calls model.eval() before accepting traffic; forgetting that single call in real ML teams has caused notorious production outages where live chatbots gave noisy, degraded answers.",
        jargonTerms: [
          { term: "Initialization", def: "Setting smart starting random numbers in weights before step 1." },
          { term: "Epoch / Step", def: "One batch update = 1 Step; one full pass through all data = 1 Epoch." },
          { term: "Dropout", def: "Randomly turning off % of neurons during training to prevent memorization." },
          { term: "model.eval()", def: "Switches off training-only behaviors (like Dropout) when testing/chatting." }
        ],
        why: "If initial weights are too large, activations explode to NaN on step 1; if too small, gradients vanish to zero. Proper initialization keeps signal variance steady across layers.",
        concepts: [
          "The canonical PyTorch training loop: forward → compute loss → zero_grad → backward → clip_grad_norm → step.",
          "Xavier / Kaiming & GPT-style scaled normal initialization (std = 0.02, scaled by 1/sqrt(2*n_layers) on residual projections).",
          "Diagnosing activation saturation and gradient norms across layers."
        ],
        code: `import torch.nn as nn

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)`,
        pitfall: "Forgetting model.eval() during validation/inference and model.train() during training—Dropout layers will otherwise randomly zero out neurons during text generation!",
        project: "Write a clean, reusable PyTorch trainer function with train/eval mode switching, gradient norm logging, and weight initialization.",
        resourceTitle: "Andrej Karpathy — Building makemore Part 3: Activations & Gradients",
        resourceNote: "Deep diagnostic masterclass on inspecting activation histograms and gradient flow.",
        links: [
          { type: "YouTube", label: "Andrej Karpathy — Building makemore Part 3: Activations & Gradients", url: "https://www.youtube.com/watch?v=P6sfmUTpUmc" },
          { type: "Community Guide", label: "Andrej Karpathy Blog — A Recipe for Training Neural Networks", url: "https://karpathy.github.io/2019/04/25/recipe/" },
          { type: "Community Guide", label: "DeepLearning.AI — Initializing Neural Networks Interactive Visualizer", url: "https://www.deeplearning.ai/ai-notes/initialization/index.html" }
        ]
      }
    ]
  },
  {
    id: "phase-3",
    number: "03",
    category: "math-pytorch",
    accent: "teal",
    duration: "Week 5",
    title: "Text Tokenization & Data Pipelines",
    subtitle: "Convert raw text corpora into subword token IDs using Byte-Pair Encoding (BPE) and construct sliding-window PyTorch DataLoaders.",
    plainBigPicture: "Neural networks can only do math on numbers, never raw English letters. A 'Tokenizer' acts like a secret decoder ring: it chops text into common chunks ('Tokens') and assigns each chunk an integer ID. Then a 'DataLoader' slices a book into millions of fill-in-the-blank training puzzles.",
    liveAiBridge: {
      headline: "Where Live Running AIs Use Phase 3 Skills",
      summary: "Every AI API bills you 'per 1 million tokens' and limits how many documents you can paste into its 'context window'—both are direct consequences of the Byte-Pair Encoding (BPE) tokenizer and special control tags you build in Phase 3.",
      examples: [
        "OpenAI Tiktoken & Llama-3 Tokenizers: GPT-4 (100k vocab) and Llama 3 (128k vocab) use the exact byte-level BPE merge algorithm you write in Module 3.3.",
        "Why LLMs Struggle to Count 'r's in 'Strawberry': Because BPE groups 'straw' + 'berry' into 2 integer IDs, the model never sees individual letters—a quirk you test firsthand in Module 3.1.",
        "ChatGPT Role Separation (<|im_start|>user): Chat apps prevent prompt injection and know when to stop replying using the exact special control tokens from Module 3.4."
      ]
    },
    jargonDecoder: [
      { term: "Token", def: "A word or common word-piece (e.g., 'ing')" },
      { term: "Tokenizer", def: "Translates text ↔ integer token IDs" },
      { term: "BPE", def: "Compression algorithm that merges frequent letter pairs" },
      { term: "Context Window", def: "Max tokens the model can read at once" },
      { term: "DataLoader", def: "Feeds shuffled training batches to the GPU" }
    ],
    achievements: [
      "Code a working Byte-Pair Encoding (BPE) tokenizer from scratch in pure Python (train, encode, and decode).",
      "Handle UTF-8 byte streams and special control tokens (<|endoftext|>, <|user|>, <|assistant|>) without crashes.",
      "Build a PyTorch sliding-window Dataset & DataLoader producing (X, Y) next-token training pairs."
    ],
    miniProject: "Code a Complete Byte-Pair Encoding (BPE) Tokenizer & Sliding-Window DataLoader",
    topics: [
      {
        id: "p3-t1",
        codeNum: "3.1",
        title: "Character-Level vs. Word-Level vs. Subword Tokenization",
        duration: "2 Days",
        plainEnglish: "If we number every single letter (a=1, b=2), sentences become super long and hard to remember. If we number every whole dictionary word, the dictionary gets huge and crashes on typos. 'Subword Tokenization' splits rare words into reusable roots like 'un-' + 'believ' + '-able'.",
        liveExample: "GitHub Copilot and Cursor generate code 3× faster than early models because their subword tokenizers merge common Python/JS indentation spaces ('    def ') into a single token ID instead of wasting 4 separate space tokens.",
        jargonTerms: [
          { term: "Character-Level", def: "1 token = 1 character (tiny vocab, very long sequences)." },
          { term: "Word-Level", def: "1 token = 1 whole word (huge vocab, fails on typos/new words)." },
          { term: "Subword Token", def: "1 token = frequent word or root/suffix chunk (~0.75 words on average)." }
        ],
        why: "Tokenization defines the atomic symbols your LLM sees and predicts—directly governing context length efficiency, spelling, and code/math accuracy.",
        concepts: [
          "Character-level: tiny vocab (~65 chars), lossless, but wastes context window on long token sequences.",
          "Word-level: huge vocab (100k+), explodes embedding matrix size, and breaks on unseen words (<UNK>).",
          "Subword tokenization (BPE / WordPiece / Unigram): best of both worlds—frequent words stay intact while rare words split into meaningful roots/suffixes."
        ],
        code: `# Inspecting how GPT-4's cl100k_base tokenizer splits text vs code:
# "tokenization" -> ["token", "ization"]  (2 tokens instead of 12 characters!)
# "   def foo():" -> ["   ", "def", " foo", "():"]`,
        pitfall: "Leading spaces matter! In GPT tokenizers, 'apple' and ' apple' have completely different token IDs—never strip trailing/leading whitespace blindly when building prompts.",
        project: "Compare sequence lengths on a 10,000-word Python + English text file across character-level, whitespace word-level, and subword tokenization.",
        resourceTitle: "Hugging Face LLM Course — Tokenizers & Karpathy's GPT Tokenizer",
        resourceNote: "Shows concrete examples of how tokenization quirks explain LLM failure modes.",
        links: [
          { type: "Community Guide", label: "OpenAI Tiktokenizer — Interactive Web Token Splitter", url: "https://tiktokenizer.vercel.app/" },
          { type: "YouTube", label: "Andrej Karpathy — Let's Build the GPT Tokenizer (Full Lecture)", url: "https://www.youtube.com/watch?v=zduSFxRajkE" },
          { type: "Blog / Docs", label: "Hugging Face LLM Course — Chapter 2.4: Tokenizers Explained", url: "https://huggingface.co/learn/llm-course/chapter2/4" }
        ]
      },
      {
        id: "p3-t2",
        codeNum: "3.2",
        title: "Unicode, UTF-8 Byte Encoding & Vocabulary Trade-offs",
        duration: "2 Days",
        plainEnglish: "Computers store text using 'UTF-8', which breaks any letter, Hindi/Tamil script, or emoji into 1 to 4 raw bytes (numbers 0–255). By starting our tokenizer with those 256 base bytes, our LLM can read literally any language or symbol on Earth without ever crashing on an 'unknown character'.",
        liveExample: "Multilingual assistants like ChatGPT and Gemini can seamlessly translate Tamil, Hindi, Japanese, Python code, and emojis in a single sentence without <UNK> errors because their vocabularies start from the exact 256 raw UTF-8 bytes from this module.",
        jargonTerms: [
          { term: "Unicode", def: "Global standard assigning a unique number to every character and emoji." },
          { term: "UTF-8 Bytes", def: "Encoding any Unicode character into 1–4 byte integers (0 to 255)." },
          { term: "<UNK> Token", def: "'Unknown' placeholder—eliminated completely by byte-level tokenizers." }
        ],
        why: "Modern tokenizers operate on raw UTF-8 bytes (0–255) rather than raw Unicode code points so every emoji, symbol, and language is representable without <UNK> tokens.",
        concepts: [
          "Unicode code points (ord('A') = 65, ord('🚀') = 128640) vs. UTF-8 variable-length byte encoding (1 to 4 bytes).",
          "Base vocabulary of 256 raw bytes guarantees 100% coverage of any text or binary string.",
          "Vocabulary size trade-offs: larger vocab (50k–128k) compresses text better (faster inference) but increases embedding/LM-head parameters."
        ],
        code: `text = "Hello 🚀 LLM"
raw_bytes = list(text.encode("utf-8"))
print("UTF-8 byte IDs:", raw_bytes)
print("Decoded back:", bytes(raw_bytes).decode("utf-8"))`,
        pitfall: "During BPE decoding, arbitrary byte sequences generated by an untrained model might form invalid UTF-8 sequences—always use .decode('utf-8', errors='replace') to avoid crashes.",
        project: "Write a helper script that inspects UTF-8 byte lengths across English, Hindi/Tamil, Python code, and emojis.",
        resourceTitle: "Andrej Karpathy — Let's Build the GPT Tokenizer (Unicode & UTF-8 Section)",
        resourceNote: "Explains why starting from UTF-8 bytes eliminates out-of-vocabulary errors forever.",
        links: [
          { type: "YouTube", label: "Andrej Karpathy — Let's Build the GPT Tokenizer (UTF-8 & Byte BPE)", url: "https://www.youtube.com/watch?v=zduSFxRajkE" },
          { type: "Community Guide", label: "Joel on Software — The Absolute Minimum Every Developer Must Know About Unicode", url: "https://www.joelonsoftware.com/2003/10/08/the-absolute-minimum-every-software-developer-absolutely-positively-must-know-about-unicode-and-character-sets-no-excuses/" },
          { type: "Blog / Docs", label: "Hugging Face LLM Course — Byte-Pair Encoding Tokenization", url: "https://huggingface.co/learn/llm-course/chapter6/5" }
        ]
      },
      {
        id: "p3-t3",
        codeNum: "3.3",
        title: "Coding Byte-Pair Encoding (BPE) from Scratch (minbpe)",
        duration: "3 Days",
        plainEnglish: "Byte-Pair Encoding (BPE) is a super simple compression trick: scan a book, find the two tokens that appear side-by-side most often (like 't' and 'h'), mint a brand-new token ID for 'th', and repeat 50,000 times. Frequent words like ' the' become a single number!",
        liveExample: "OpenAI's open-source 'tiktoken' library and Meta's Llama-3 tokenizer execute the exact merge_pair() lookup table you build here on every API request before a single GPU layer runs.",
        jargonTerms: [
          { term: "BPE (Byte-Pair Encoding)", def: "Iteratively merging the most frequent adjacent token pair into one new token." },
          { term: "Merge Table", def: "The ordered rulebook of pair merges learned during tokenizer training." },
          { term: "encode() / decode()", def: "Converting text → token IDs (encode) and token IDs → text (decode)." }
        ],
        why: "BPE is the exact algorithm used to train GPT-2, GPT-4, and Llama tokenizers. Coding it in ~60 lines of Python cements how compression meets NLP.",
        concepts: [
          "Counting all adjacent token pairs across the corpus with a frequency dictionary.",
          "Merging the highest-frequency pair (p0, p1) into a new token ID (256, 257, ...) and storing the merge rule.",
          "Encoding new text by repeatedly applying learned merges in priority order, and decoding via vocab lookup."
        ],
        code: `def merge_pair(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out`,
        pitfall: "In encode(), you must apply the merge with the lowest merge index (earliest learned merge) at each step—not arbitrary dictionary iteration order!",
        project: "Build a BasicTokenizer class with train(text, vocab_size), encode(text), and decode(ids), train a 512-token vocab on TinyShakespeare, and print the top 30 merges.",
        resourceTitle: "Andrej Karpathy — minbpe Repository & Tokenizer Lecture",
        resourceNote: "Walks through BasicTokenizer and RegexTokenizer step by step.",
        links: [
          { type: "YouTube", label: "Andrej Karpathy — Let's Build the GPT Tokenizer", url: "https://www.youtube.com/watch?v=zduSFxRajkE" },
          { type: "Community Guide", label: "Andrej Karpathy — Official minbpe GitHub Repository", url: "https://github.com/karpathy/minbpe" },
          { type: "Community Guide", label: "GeeksforGeeks — Byte Pair Encoding (BPE) in NLP Step-by-Step", url: "https://www.geeksforgeeks.org/byte-pair-encoding-bpe-in-nlp/" }
        ]
      },
      {
        id: "p3-t4",
        codeNum: "3.4",
        title: "Special Tokens (<|endoftext|>, PAD, BOS/EOS) & Context Windows",
        duration: "2 Days",
        plainEnglish: "How does an AI know where one Wikipedia article ends and another begins, or when it should stop replying to your chat message? We inject special 'control tag' tokens like <|endoftext|> into the vocabulary that act like punctuation marks for conversation boundaries.",
        liveExample: "When you hit Send in ChatGPT or Ollama, the UI silently wraps your message in special tokens like <|im_start|>user ... <|im_end|><|im_start|>assistant—and the streaming reply stops the exact millisecond the model emits the <|im_end|> EOS token ID.",
        jargonTerms: [
          { term: "Special Tokens", def: "Reserved control IDs like <|endoftext|> or <|assistant|> never split by BPE." },
          { term: "EOS (End of Sequence)", def: "Stop-sign token telling the LLM 'my answer is finished, stop generating'." },
          { term: "Context Window", def: "The maximum token buffer (e.g., 256 or 8,192) the model can see at once." }
        ],
        why: "Special control tokens delimit separate documents during pretraining and separate User vs. Assistant turns during chat fine-tuning.",
        concepts: [
          "Document separators (<|endoftext|>) preventing unrelated articles from bleeding into one another.",
          "Conversation delimiters (<|im_start|>, <|im_end|>, <|user|>, <|assistant|>) and padding tokens (<|pad|>).",
          "Context length (block_size / max_seq_len): the hard upper limit on tokens the model can attend to simultaneously."
        ],
        code: `special_tokens = {
    "<|endoftext|>": 50256, # Marks boundary between unrelated documents
    "<|user|>": 50257,      # Marks start of human prompt
    "<|assistant|>": 50258  # Marks start of AI reply
}`,
        pitfall: "Allowing regular BPE regex rules to split '<|endoftext|>' into '<', '|', 'endoftext', '|', '>'—special tokens must be matched and mapped atomically before BPE runs.",
        project: "Extend your BPE tokenizer to register and encode/decode custom special tokens like <|endoftext|> and <|pad|>.",
        resourceTitle: "Sebastian Raschka — Chapter 2: Working with Text Data & Special Tokens",
        resourceNote: "Shows practical examples of handling unknown words and document boundary markers.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Chapter 2 Code Notebooks (LLMs-from-scratch)", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02" },
          { type: "YouTube", label: "Sebastian Raschka — Build an LLM from Scratch Playlist", url: "https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11" },
          { type: "Blog / Docs", label: "Hugging Face Docs — Chat Templates & Special Tokens", url: "https://huggingface.co/docs/transformers/main/en/chat_templating" }
        ]
      },
      {
        id: "p3-t5",
        codeNum: "3.5",
        title: "Sliding-Window Input/Target Pairs (X, Y) & PyTorch Dataset/DataLoader",
        duration: "2 Days",
        plainEnglish: "LLMs learn via self-supervised fill-in-the-blank: we don't need human labelers! We take any raw text sequence like [10, 20, 30, 40], set Input X = [10, 20, 30], and shift Target Y right by one slot to [20, 30, 40]. That single chunk gives 3 next-word prediction exercises at once!",
        liveExample: "Foundation labs pack trillions of tokens from CommonCrawl, GitHub, and Wikipedia into fixed-length context chunks (e.g., 8,192 tokens) with Y shifted by 1 index—enabling a single GPU pass to grade 8,192 next-word predictions simultaneously.",
        jargonTerms: [
          { term: "Self-Supervised", def: "Training directly on raw unlabeled text by hiding the next word as the answer." },
          { term: "Input / Target (X, Y)", def: "Y is simply X shifted forward by 1 token (the 'next token' answer key)." },
          { term: "Stride", def: "How many tokens the sliding window jumps forward to create the next sample." },
          { term: "Dataset & DataLoader", def: "PyTorch utilities that package samples and serve shuffled batches." }
        ],
        why: "A single sequence of T+1 tokens yields T self-supervised next-token prediction training examples in parallel.",
        concepts: [
          "Shifting targets by 1 index: X = data[i : i+T], Y = data[i+1 : i+T+1].",
          "Stride parameter: overlapping windows vs. non-overlapping chunks (stride = T) to prevent overfitting.",
          "Wrapping token arrays in torch.utils.data.Dataset and batching with DataLoader(batch_size, shuffle=True)."
        ],
        code: `import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, token_ids, max_len, stride):
        self.inputs, self.targets = [], []
        for i in range(0, len(token_ids) - max_len, stride):
            self.inputs.append(torch.tensor(token_ids[i : i + max_len]))
            self.targets.append(torch.tensor(token_ids[i + 1 : i + max_len + 1]))
    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]`,
        pitfall: "Setting stride=1 on a large corpus creates near-duplicate overlapping windows that cause rapid memorization/overfitting; use stride = max_len for pretraining.",
        project: "Create a working GPTDataset and DataLoader over TinyShakespeare and verify batch shapes X: (32, 128) and Y: (32, 128).",
        resourceTitle: "PyTorch Datasets & DataLoaders + Sebastian Raschka Ch. 2.6",
        resourceNote: "Includes clear diagrams of input-target shifting and DataLoader batching.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Sliding Window DataLoader Code", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02" },
          { type: "YouTube", label: "Vizuara — Building LLMs from Scratch Playlist (Data & Tokenization)", url: "https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu" },
          { type: "Blog / Docs", label: "PyTorch Official Tutorials — Datasets & DataLoaders", url: "https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html" }
        ]
      }
    ]
  },
  {
    id: "phase-4",
    number: "04",
    category: "transformer",
    accent: "indigo",
    duration: "Week 6–8",
    title: "Transformer Architecture & Self-Attention (The Heart of GPT)",
    subtitle: "Build every internal component of a Decoder-Only Transformer from scratch in pure PyTorch—from embeddings and QKV causal attention to stacked GPT blocks.",
    plainBigPicture: "This is the core engine behind ChatGPT, Claude, and Llama. First, token IDs become rich meaning vectors ('Embeddings'). Next, 'Self-Attention' lets words look back at earlier words for context (so 'it' knows whether you meant 'the trophy' or 'the suitcase'). Stacking these blocks creates a GPT!",
    liveAiBridge: {
      headline: "Where Live Running AIs Use Phase 4 Skills",
      summary: "The Decoder-Only Transformer block you assemble in Phase 4 is the exact architectural blueprint running inside ChatGPT, Claude, Llama 3, Qwen, and DeepSeek—scaled up with more layers and wider embeddings.",
      examples: [
        "128K Long-Context Windows (Claude & Gemini): Powered by the RoPE positional rotations (Module 4.1) and Multi-Head Attention (Module 4.4) you implement here.",
        "Real-Time Streaming Word-by-Word Output: Possible only because of the Causal Lower-Triangular Mask (Module 4.3), which ensures past tokens never change when new tokens are generated.",
        "Deep 80–96 Layer Reasoning Models: Stay numerically stable from layer 1 to layer 96 thanks to the Pre-Norm + Residual Skip Connections you code in Module 4.5."
      ]
    },
    jargonDecoder: [
      { term: "Embedding", def: "List of numbers capturing a word's meaning" },
      { term: "Self-Attention", def: "Words looking back at earlier context words" },
      { term: "Q, K, V", def: "Query (question), Key (label), Value (info)" },
      { term: "Causal Mask", def: "Blinder hiding future words during training" },
      { term: "LayerNorm", def: "Keeps numbers balanced across deep layers" }
    ],
    achievements: [
      "Implement Scaled Dot-Product Self-Attention and Multi-Head Attention with causal lower-triangular masking from scratch.",
      "Combine Token Embeddings, Positional Encodings (Learned & RoPE), LayerNorm, Residual Connections, and Feed-Forward MLPs.",
      "Assemble a complete, parameter-verified Decoder-Only GPT model in pure PyTorch ready for pretraining."
    ],
    miniProject: "Assemble a Complete Decoder-Only GPT Model in Pure PyTorch & Verify Forward Pass",
    topics: [
      {
        id: "p4-t1",
        codeNum: "4.1",
        title: "Token Embeddings & Positional Encodings (Learned vs. Sinusoidal vs. RoPE)",
        duration: "3 Days",
        plainEnglish: "Token ID #4881 ('France') is just an arbitrary serial number—ID #4882 isn't 'similar to France'. An 'Embedding' translates each ID into a learned list of 256+ numbers where similar concepts sit close together. Because attention doesn't naturally know word order, we also add a 'Positional Encoding' stamp (Position 0, 1, 2...).",
        liveExample: "Llama 3, Mistral, Qwen, and Gemma all use Rotary Position Embeddings (RoPE) from this module—allowing engineers to stretch a model's memory from 8,192 tokens out to 128,000 tokens so you can paste an entire codebase into one chat.",
        jargonTerms: [
          { term: "nn.Embedding", def: "A lookup table translating integer token IDs into rich decimal vectors." },
          { term: "Positional Embedding", def: "Order signal added to tokens so 'dog bites man' ≠ 'man bites dog'." },
          { term: "RoPE", def: "Rotary Position Embedding—rotates vectors by position angle (used in Llama)." }
        ],
        why: "Self-attention treats inputs as an unordered set; positional encodings inject word-order geometry so the model knows 'A before B' differs from 'B before A'.",
        concepts: [
          "nn.Embedding(vocab_size, emb_dim): a differentiable lookup table mapping integer IDs to dense vectors.",
          "Absolute Learned Positional Embeddings (GPT-2: tok_emb + pos_emb) vs. Sinusoidal waves (Vaswani et al. 2017).",
          "Rotary Position Embeddings (RoPE): rotating Query and Key vectors by angle m·θ so dot products naturally encode relative token distance (Llama/Qwen)."
        ],
        code: `import torch, torch.nn as nn

B, T, vocab_size, emb_dim, max_len = 2, 16, 50257, 256, 512
tok_emb = nn.Embedding(vocab_size, emb_dim)
pos_emb = nn.Embedding(max_len, emb_dim)

idx = torch.randint(0, vocab_size, (B, T))
x = tok_emb(idx) + pos_emb(torch.arange(T))`,
        pitfall: "Passing sequences longer than max_len into a learned positional embedding table—torch.arange(T) will throw an IndexError if T > max_len!",
        project: "Plot the cosine similarity matrix of a learned positional embedding table before and after training to observe local neighborhood patterns.",
        resourceTitle: "Sebastian Raschka — Positional Embeddings Blog & Jay Alammar GPT-2",
        resourceNote: "Compare additive GPT-2 positional embeddings with modern multiplicative RoPE rotations.",
        links: [
          { type: "Community Guide", label: "Jay Alammar — The Illustrated GPT-2 (Embeddings & Decoder)", url: "https://jalammar.github.io/illustrated-gpt2/" },
          { type: "Community Guide", label: "EleutherAI Blog — Rotary Embeddings (RoPE): A Relative Revolution", url: "https://blog.eleuther.ai/rotary-embeddings/" },
          { type: "YouTube", label: "3Blue1Brown — Transformers, the tech behind LLMs (Embeddings)", url: "https://www.youtube.com/watch?v=wjZofJX0v4M" }
        ]
      },
      {
        id: "p4-t2",
        codeNum: "4.2",
        title: "Query, Key, Value (QKV) Projections & Scaled Dot-Product Attention",
        duration: "4 Days",
        plainEnglish: "Think of Self-Attention like searching YouTube: every word emits a 'Query' (what I am looking for, like 'who performed this action?'), a 'Key' (what label I advertise, like 'I am a noun/subject'), and a 'Value' (the actual information I share if matched). High Query·Key match = strong attention!",
        liveExample: "When you ask ChatGPT 'The trophy didn't fit in the suitcase because it was too big—what was too big?', QKV attention scores link 'it' strongly back to 'trophy' instead of 'suitcase' by matching semantic Queries and Keys.",
        jargonTerms: [
          { term: "Query (Q)", def: "Vector representing what context a token is currently searching for." },
          { term: "Key (K)", def: "Vector advertising what grammatical/semantic role a token holds." },
          { term: "Value (V)", def: "The actual content vector passed forward when Q matches K." },
          { term: "Attention Weights", def: "Percentages (summing to 100%) showing how much each past word matters." }
        ],
        why: "Self-Attention lets each token dynamically query earlier tokens for context—e.g., resolving what pronoun 'it' refers to in a paragraph.",
        concepts: [
          "Linear projections W_q, W_k, W_v transforming input x into Queries ('what I seek'), Keys ('what I advertise'), and Values ('what I pass on').",
          "Raw affinity scores: scores = Q @ K.transpose(-2, -1) of shape (B, T, T).",
          "Scaling by 1 / sqrt(head_dim): keeping dot-product variance near 1 so Softmax doesn't saturate into near-zero gradients."
        ],
        code: `import torch, torch.nn.functional as F

B, T, d_k = 2, 6, 32
Q, K, V = torch.randn(B, T, d_k), torch.randn(B, T, d_k), torch.randn(B, T, d_k)

scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5) # Shape: (B, T, T)
attn_weights = F.softmax(scores, dim=-1)
context = attn_weights @ V                         # Shape: (B, T, d_k)`,
        pitfall: "Forgetting to divide by sqrt(d_k)! Without scaling, large embedding dimensions produce huge dot products that push Softmax into one-hot spikes with near-zero gradients.",
        project: "Implement a single-head SelfAttention class in PyTorch and verify that every row of attn_weights sums to 1.0.",
        resourceTitle: "3Blue1Brown — Attention in Transformers, Visually Explained",
        resourceNote: "Best geometric animation on the internet for understanding Q, K, V dot products.",
        links: [
          { type: "YouTube", label: "3Blue1Brown — Attention in Transformers, Step-by-Step (Ch. 6)", url: "https://www.youtube.com/watch?v=eMlx5fFNoYc" },
          { type: "Community Guide", label: "Harvard NLP — The Annotated Transformer (PyTorch Line-by-Line)", url: "https://nlp.seas.harvard.edu/annotated-transformer/" },
          { type: "Community Guide", label: "Peter Bloem's Blog — Transformers From Scratch (Deep Math & Code)", url: "https://peterbloem.nl/blog/transformers" }
        ]
      },
      {
        id: "p4-t3",
        codeNum: "4.3",
        title: "Causal Masking (tril) for Autoregressive Next-Token Prediction",
        duration: "2 Days",
        plainEnglish: "During training, we feed an entire sentence into the GPU at once for speed. Without a 'Causal Mask', word #3 could cheat by peeking ahead at word #4! We cover the upper-right triangle of the attention grid with negative infinity (-inf) so future words get exactly 0% attention.",
        liveExample: "Causal masking is the exact mathematical property that enables KV-Caching in ChatGPT and Ollama—because earlier tokens cannot see future tokens, their Key and Value vectors never change and can be cached in GPU RAM for instant typing speed.",
        jargonTerms: [
          { term: "Causal Mask", def: "Triangular barrier blocking tokens from peeking at future tokens." },
          { term: "torch.tril", def: "'Triangle Lower'—keeps the bottom-left triangle and zeroes out top-right." },
          { term: "register_buffer", def: "Stores fixed non-learnable tensors (like masks) inside a PyTorch module." }
        ],
        why: "During training, we feed all T tokens at once for speed. Causal masking prevents token t from peeking at future tokens t+1..T so training matches inference.",
        concepts: [
          "Lower-triangular matrix torch.tril(torch.ones(T, T)) where upper-right future entries are 0.",
          "Masking future positions with -inf BEFORE Softmax so exp(-inf) becomes exactly 0.0 probability.",
          "Registering the mask via self.register_buffer('mask', ...) so it moves to GPU automatically without being treated as a trainable parameter."
        ],
        code: `T = 4
scores = torch.randn(T, T)
mask = torch.tril(torch.ones(T, T))
masked_scores = scores.masked_fill(mask == 0, float('-inf'))
print(torch.softmax(masked_scores, dim=-1))`,
        pitfall: "Masking with 0 AFTER Softmax instead of -inf BEFORE Softmax! Masking after Softmax breaks row probability sums (rows won't sum to 1 anymore).",
        project: "Add causal masking with register_buffer to your SelfAttention module and print the lower-triangular attention heatmap.",
        resourceTitle: "Andrej Karpathy — Let's Build GPT (The Mathematical Trick in Self-Attention)",
        resourceNote: "Walks through weighted aggregation of past tokens using lower-triangular matrices.",
        links: [
          { type: "YouTube", label: "Andrej Karpathy — Let's build GPT: from scratch, in code, spelled out", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY" },
          { type: "Community Guide", label: "Jay Alammar — The Illustrated GPT-2 (Masked Self-Attention)", url: "https://jalammar.github.io/illustrated-gpt2/#part-2-the-illustrated-self-attention" },
          { type: "Community Guide", label: "Sebastian Raschka — Understanding & Coding Self-Attention From Scratch", url: "https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html" }
        ]
      },
      {
        id: "p4-t4",
        codeNum: "4.4",
        title: "Multi-Head Self-Attention & Tensor Reshaping / Concatenation",
        duration: "4 Days",
        plainEnglish: "Instead of one spotlight looking at the sentence, 'Multi-Head Attention' splits the embedding into 6 to 32 parallel spotlights ('Heads'). Head 1 might track grammar/verbs, Head 2 tracks pronouns ('it' → 'cat'), and Head 3 tracks punctuation—then all heads merge their findings back together.",
        liveExample: "In GPT-3 (96 heads) and Llama-3-70B (64 heads), mechanistic interpretability tools like BertViz show specialized heads in action: one head tracks matching Python parentheses '()', while another tracks subject-verb agreement across clauses.",
        jargonTerms: [
          { term: "Attention Head", def: "An independent QKV spotlight focusing on one type of word relationship." },
          { term: "head_dim", def: "Width of each head (emb_dim divided by n_heads, e.g., 384 / 6 = 64)." },
          { term: ".contiguous()", def: "Re-aligns tensor memory layout after swapping axes so .view() works." }
        ],
        why: "Multiple heads let the model attend simultaneously to syntactic structure, entity references, and local phrasing across distinct subspaces.",
        concepts: [
          "Splitting C = n_heads * head_dim (e.g., 384 = 6 heads × 64 dims).",
          "Batched 4D tensor trick: projecting Q, K, V at once, reshaping to (B, T, n_heads, head_dim), and transposing to (B, n_heads, T, head_dim).",
          "Merging heads back via .transpose(1, 2).contiguous().view(B, T, C) followed by output projection W_o."
        ],
        code: `import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.n_heads, self.head_dim = n_heads, emb_dim // n_heads
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        return self.proj(q.transpose(1, 2).contiguous().view(B, T, C))`,
        pitfall: "Forgetting .contiguous() before .view(B, T, C) after .transpose(1, 2)—PyTorch will raise a RuntimeError because transposed tensors have non-contiguous memory strides.",
        project: "Code a complete batched MultiHeadAttention module in PyTorch and verify that it matches the output of n_heads separate single-head modules.",
        resourceTitle: "Sebastian Raschka — Multi-Head Attention & Vizuara SLM Walkthrough",
        resourceNote: "Compares naive stacked heads with efficient batched weight splits.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Chapter 3.5 Multi-Head Attention Code", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03" },
          { type: "YouTube", label: "Vizuara — Build a Small Language Model (SLM) From Scratch", url: "https://www.youtube.com/watch?v=pOFcwcwtv3k" },
          { type: "Community Guide", label: "Lilian Weng (OpenAI) — Attention? Attention! Technical Blog", url: "https://lilianweng.github.io/posts/2018-06-24-attention/" }
        ]
      },
      {
        id: "p4-t5",
        codeNum: "4.5",
        title: "LayerNorm / RMSNorm, Residual Connections & Feed-Forward MLP Blocks",
        duration: "3 Days",
        plainEnglish: "When you stack 12 or 96 Transformer blocks, numbers can drift wildly out of scale and learning signals get lost. 'LayerNorm' re-centers every token's numbers to a clean bell curve, and 'Residual Connections' (x = x + layer(x)) act like an express elevator carrying original info and gradients cleanly top-to-bottom.",
        liveExample: "Llama 3, Gemma, and Qwen replace standard LayerNorm with RMSNorm (dropping mean subtraction for ~15% faster GPU kernels) inside the exact Pre-Norm residual block structure you build in this module.",
        jargonTerms: [
          { term: "LayerNorm / RMSNorm", def: "Re-scales each token's numbers to mean 0, std 1 so deep layers stay stable." },
          { term: "Residual Connection", def: "Adding the input directly to the output (x + block(x)) as a gradient highway." },
          { term: "Transformer Block", def: "One complete unit: Norm → Attention → Add → Norm → MLP → Add." }
        ],
        why: "Deep Transformers (12 to 96+ layers) only train stably because residual highways let gradients flow unimpeded from loss straight back to early embeddings.",
        concepts: [
          "Pre-Norm Transformer Block: x = x + Attention(Norm(x)), followed by x = x + MLP(Norm(x)).",
          "LayerNorm (normalizing across the C feature dimension per token) vs. RMSNorm (faster root-mean-square scaling used in Llama).",
          "Position-wise Feed-Forward MLP: expanding C → 4*C → C with GELU activation and dropout."
        ],
        code: `class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.emb_dim)
        self.attn = MultiHeadAttention(cfg.emb_dim, cfg.n_heads)
        self.ln2 = nn.LayerNorm(cfg.emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x`,
        pitfall: "Using BatchNorm instead of LayerNorm/RMSNorm! BatchNorm averages across the batch dimension, which breaks variable-length sequences and single-token autoregressive inference.",
        project: "Implement LayerNorm from scratch using x.mean(dim=-1, keepdim=True) and x.var(dim=-1, keepdim=True) and verify it matches nn.LayerNorm to 1e-5.",
        resourceTitle: "Sebastian Raschka — Chapter 4: Implementing a GPT Model from Scratch",
        resourceNote: "Implements LayerNorm, GELU, shortcut connections, and the full TransformerBlock cleanly.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Chapter 4 TransformerBlock Code", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04" },
          { type: "YouTube", label: "Andrej Karpathy — Let's build GPT (Residual Connections & LayerNorm)", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY" },
          { type: "Community Guide", label: "Pinecone Learning Center — Layer Normalization in Transformers Explained", url: "https://www.pinecone.io/learn/batch-layer-normalization/" }
        ]
      },
      {
        id: "p4-t6",
        codeNum: "4.6",
        title: "Assembling the Full Decoder-Only GPT Architecture & Parameter Counting",
        duration: "3 Days",
        plainEnglish: "Now we snap all the Lego bricks together! Token IDs enter the bottom, get stamped with positional embeddings, pass through N stacked Transformer Blocks, and exit through a final 'LM Head' linear layer that scores all 50,257 possible next words. 'Weight Tying' shares the input and output word tables to save memory.",
        liveExample: "Google's Gemma-2B/9B and Meta's Llama-3.2-1B/3B models use the exact weight-tying trick (tok_emb.weight = lm_head.weight) from this module—saving over 250 million parameters on large vocabularies so the model fits comfortably on consumer laptops and phones.",
        jargonTerms: [
          { term: "Decoder-Only GPT", def: "Autoregressive Transformer predicting one next token at a time (GPT/Llama)." },
          { term: "nn.ModuleList", def: "PyTorch's official list wrapper so stacked layers register properly for GPU & training." },
          { term: "LM Head", def: "Final linear projection turning token vectors into 50,257 vocabulary logits." },
          { term: "Weight Tying", def: "Reusing the input embedding matrix as the output LM Head to save millions of params." },
          { term: "Parameters", def: "Total count of learnable numbers in the model (e.g., 124M in GPT-2 Small)." }
        ],
        why: "Connecting embeddings, N Transformer blocks, final LayerNorm, and the Language Model Head gives you a genuine GPT architecture ready for pretraining.",
        concepts: [
          "Stacking N identical TransformerBlocks via nn.ModuleList.",
          "Final LayerNorm + Linear LM Head projecting (B, T, C) → (B, T, vocab_size).",
          "Weight tying (tok_emb.weight = lm_head.weight) to save millions of parameters and improve generalization."
        ],
        code: `class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.emb_dim)
        self.lm_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight`,
        pitfall: "Storing Transformer blocks in a plain Python list (self.blocks = [...]) instead of nn.ModuleList—PyTorch will silently ignore their weights during model.parameters() and .to(device)!",
        project: "Instantiate a ~10M-parameter mini-GPT (6 layers, 6 heads, 384 emb_dim) and a 124M GPT-2 Small config, printing exact parameter counts per component.",
        resourceTitle: "Andrej Karpathy nanoGPT & freeCodeCamp Full LLM Build",
        resourceNote: "Complete end-to-end GPTModel walkthrough with parameter breakdown.",
        links: [
          { type: "Community Guide", label: "Andrej Karpathy — Official nanoGPT Repository (model.py)", url: "https://github.com/karpathy/nanoGPT" },
          { type: "YouTube", label: "freeCodeCamp — Create a Large Language Model from Scratch with Python", url: "https://www.youtube.com/watch?v=UU1WVnMk4E8" },
          { type: "Community Guide", label: "Brendan Bycroft — LLM Visualization (Interactive 3D GPT Walkthrough)", url: "https://bbycroft.net/llm" }
        ]
      }
    ]
  },
  {
    id: "phase-5",
    number: "05",
    category: "training",
    accent: "amber",
    duration: "Week 9–10",
    title: "Pretraining, Evaluation & Text Generation",
    subtitle: "Train your custom GPT from random weights on real text, track loss & perplexity curves, and decode coherent completions.",
    plainBigPicture: "A freshly built GPT starts with random weights—so its first output is pure keyboard-smash gibberish. During 'Pretraining', we feed it Shakespeare or stories, measure its surprise ('Perplexity'), gently warm up and cool down its learning speed, and watch coherent English emerge!",
    liveAiBridge: {
      headline: "Where Live Running AIs Use Phase 5 Skills",
      summary: "Every AI API parameter you configure in production apps—temperature, top_p, max_tokens—and every training stability safeguard used to train frontier models comes straight from Phase 5.",
      examples: [
        "OpenAI / Anthropic API Parameters: Setting temperature=0.0 for deterministic JSON extraction or top_p=0.9 for creative writing runs the exact sampling filter you code in Module 5.5.",
        "Gradient Accumulation & Clipping (Module 5.2): Used on every large GPU cluster to reach multi-million-token batch sizes without exploding gradients.",
        "Cosine LR Warmup & Decay (Module 5.3): The universal learning-rate curve shown in Meta's Llama and DeepSeek technical reports."
      ]
    },
    jargonDecoder: [
      { term: "Pretraining", def: "Teaching a random model language from raw text" },
      { term: "Perplexity", def: "How confused the model is (1 = certain)" },
      { term: "Overfitting", def: "Memorizing training text instead of generalizing" },
      { term: "Temperature", def: "Creativity dial when picking the next word" },
      { term: "Top-k / Top-p", def: "Filtering out weird low-probability words" }
    ],
    achievements: [
      "Pretrain your custom GPT from random weights on real text using gradient accumulation, gradient clipping, and cosine LR warmup.",
      "Measure model uncertainty via Perplexity and diagnose overfitting vs. underfitting from train/validation loss curves.",
      "Code an autoregressive text generator supporting Greedy, Temperature scaling, Top-k, and Top-p (Nucleus) sampling."
    ],
    miniProject: "Pretrain a Mini-GPT on TinyShakespeare & Generate Text with Temperature/Top-k/Top-p",
    topics: [
      {
        id: "p5-t1",
        codeNum: "5.1",
        title: "Next-Token Prediction Objective & Perplexity Metric",
        duration: "2 Days",
        plainEnglish: "Is a Cross-Entropy loss of 2.3 good or bad? Taking e^2.3 ≈ 10 gives 'Perplexity'—which means the model is roughly as uncertain as rolling a 10-sided die to guess the next word. Lower perplexity = smarter, less surprised language model!",
        liveExample: "When companies compress an LLM from 16-bit down to 4-bit GGUF for Ollama, engineers check Perplexity on WikiText-2 before and after compression—ensuring perplexity barely moves (e.g., 5.68 → 5.74) so intelligence stays intact.",
        jargonTerms: [
          { term: "Perplexity", def: "exp(loss)—effective number of equally likely next words the model considers." },
          { term: "Validation Split", def: "Held-out 10% of text never shown during training, used for honest grading." },
          { term: "torch.no_grad()", def: "Disables gradient tracking during evaluation to save 50% RAM and run faster." }
        ],
        why: "Understanding how 3D logits (B, T, V) flatten into 2D cross-entropy inputs—and how Perplexity translates loss into human-readable 'branching factor'—is core to LLM evaluation.",
        concepts: [
          "Flattening logits from (B, T, vocab_size) → (B*T, vocab_size) and targets from (B, T) → (B*T,) for F.cross_entropy.",
          "Perplexity = exp(loss): a perplexity of 15 means the model is roughly as uncertain as choosing uniformly among 15 plausible next words.",
          "Evaluating loss on a held-out validation split with torch.no_grad()."
        ],
        code: `B, T, V = logits.shape
loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
perplexity = torch.exp(loss)
print(f"Loss: {loss.item():.3f} | Perplexity: {perplexity.item():.2f}")`,
        pitfall: "Computing validation loss with gradients enabled—always wrap evaluation loops in 'with torch.no_grad():' to cut memory usage in half and double speed.",
        project: "Write an evaluate_loss(model, val_loader) helper that reports average validation loss and perplexity over 50 batches.",
        resourceTitle: "Hugging Face Transformers Docs — Perplexity of Fixed-Length Models",
        resourceNote: "Explains the mathematical link between Cross-Entropy loss and Perplexity.",
        links: [
          { type: "Community Guide", label: "Chip Huyen — Evaluation Metrics & Perplexity Intuition in Language Modeling", url: "https://huyenchip.com/2023/10/10/multimodal.html" },
          { type: "Community Guide", label: "Sebastian Raschka — Chapter 5 Pretraining & Evaluation Code", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05" },
          { type: "Blog / Docs", label: "Hugging Face Docs — Perplexity of Fixed-Length Models", url: "https://huggingface.co/docs/transformers/perplexity" }
        ]
      },
      {
        id: "p5-t2",
        codeNum: "5.2",
        title: "Pretraining Loop on TinyShakespeare: Gradient Accumulation & Clipping",
        duration: "4 Days",
        plainEnglish: "Your laptop GPU might only fit 8 sentences in memory at once, but LLMs train better when averaging 64 sentences per update. 'Gradient Accumulation' runs 8 small mini-batches and adds up their gradients before stepping! 'Gradient Clipping' acts like a speed limiter so one weird sentence never blows up your weights.",
        liveExample: "Meta trained Llama 3 using a global batch size of 4 million tokens per optimizer step by combining gradient accumulation across GPUs with max_norm=1.0 gradient clipping—preventing corrupted web documents from crashing multi-week runs.",
        jargonTerms: [
          { term: "Gradient Accumulation", def: "Summing gradients across several mini-batches to simulate a big batch." },
          { term: "Gradient Clipping", def: "Capping giant gradient spikes (max_norm=1.0) to prevent training crashes." },
          { term: "Checkpoint (.pt)", def: "Saved snapshot of model + optimizer weights so you can resume anytime." }
        ],
        why: "Real LLMs train with large effective batch sizes for smooth gradients; gradient accumulation lets your laptop/GPU simulate large batches without running out of RAM.",
        concepts: [
          "Gradient accumulation: dividing loss by grad_accum_steps, calling loss.backward() across micro-batches, and stepping optimizer once.",
          "Gradient norm clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) to prevent sudden loss spikes.",
          "Checkpointing model + optimizer state dicts (torch.save) every N steps."
        ],
        code: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, "ckpt.pt")`,
        pitfall: "Forgetting to divide loss by grad_accum_steps before calling loss.backward()—otherwise your effective learning rate scales up by grad_accum_steps!",
        project: "Pretrain your mini-GPT on TinyShakespeare (or TinyStories) for 3,000 steps and watch validation loss drop from ~10.8 down below ~1.6.",
        resourceTitle: "Andrej Karpathy — nanoGPT Training Loop & TinyShakespeare Dataset",
        resourceNote: "Shows real-time loss progression and sample text evolving from gibberish to Shakespearean dialogue.",
        links: [
          { type: "Community Guide", label: "Andrej Karpathy — nanoGPT train.py (Gradient Accumulation & Checkpoints)", url: "https://github.com/karpathy/nanoGPT/blob/master/train.py" },
          { type: "Community Guide", label: "Hugging Face Ultra-Scale Playbook — Gradient Accumulation & Memory Math", url: "https://huggingface.co/spaces/nanotron/ultrascale-playbook" },
          { type: "YouTube", label: "Andrej Karpathy — Zero to Hero Playlist (Reproducing GPT-2)", url: "https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ" }
        ]
      },
      {
        id: "p5-t3",
        codeNum: "5.3",
        title: "Learning Rate Warmup & Cosine Decay Scheduling",
        duration: "2 Days",
        plainEnglish: "Starting training at full speed when weights are totally random is like flooring a cold car engine. 'Linear Warmup' gently ramps up the learning rate over the first few hundred steps, and 'Cosine Decay' smoothly slows down near the finish line so the model parks cleanly in the lowest valley.",
        liveExample: "Every Llama, Mistral, and Qwen pretraining run publishes a cosine learning-rate curve starting with a 2,000-step linear warmup and decaying down to 10% of peak LR—the exact formula implemented in your get_lr() function.",
        jargonTerms: [
          { term: "LR Warmup", def: "Gradually increasing learning rate from 0 to peak over early steps." },
          { term: "Cosine Decay", def: "Smoothly lowering learning rate along a gentle wave toward the end of training." }
        ],
        why: "At step 0, AdamW's variance estimates are noisy and weights are random—warming up the learning rate gradually avoids early instability, while cosine decay settles into sharp minima.",
        concepts: [
          "Linear warmup phase (first ~5% of steps): ramping LR smoothly from 0 up to peak_lr.",
          "Cosine annealing decay: lowering LR smoothly down to min_lr (typically 10% of peak_lr).",
          "Updating param_group['lr'] dynamically each step."
        ],
        code: `import math

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))`,
        pitfall: "Leaving peak learning rate constant for the entire run—without decay, loss bounces noisily around the minimum instead of converging cleanly.",
        project: "Plot your get_lr() schedule curve in matplotlib across 5,000 steps and integrate it into your pretraining loop.",
        resourceTitle: "Sebastian Raschka — Appendix D: Learning Rate Warmup & Cosine Decay",
        resourceNote: "Covers learning rate warmup, cosine decay, and gradient clipping in detail.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Appendix D Training Bells & Whistles Code", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-D" },
          { type: "Community Guide", label: "MachineLearningMastery — Learning Rate Warmup & Cosine Annealing Guide", url: "https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/" },
          { type: "YouTube", label: "Vizuara — Building LLMs from Scratch Playlist (Training Tricks)", url: "https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu" }
        ]
      },
      {
        id: "p5-t4",
        codeNum: "5.4",
        title: "Diagnosing Overfitting, Underfitting & Loss Curves",
        duration: "2 Days",
        plainEnglish: "Imagine a student who memorizes exact practice exam answers ('Overfitting') vs. a student who barely studied at all ('Underfitting'). Plotting both Training Loss (practice exam) and Validation Loss (unseen final exam) reveals immediately whether your model is memorizing or genuinely learning patterns.",
        liveExample: "Production ML teams monitor live train vs. validation loss dashboards in MLflow or Weights & Biases; if validation loss diverges upward during domain fine-tuning, automated early-stopping halts training before the bot starts parroting training data verbatim.",
        jargonTerms: [
          { term: "Overfitting", def: "Train loss drops near 0 while validation loss rises (pure memorization)." },
          { term: "Underfitting", def: "Both train and validation loss stay high (model too small or undertrained)." },
          { term: "MLflow", def: "Local dashboard tool from your setup list for logging loss charts & runs." }
        ],
        why: "Reading train vs. validation loss curves tells you immediately whether your model needs more capacity, more data, stronger regularization, or earlier stopping.",
        concepts: [
          "Healthy convergence: both train and validation loss decrease smoothly together.",
          "Overfitting: train loss keeps falling while validation loss turns upward (fix with dropout, weight decay, or more data).",
          "Underfitting: both losses plateau high (fix by increasing model width/layers, training longer, or tuning LR)."
        ],
        code: `import mlflow
# mlflow.log_metric("train_loss", train_loss, step=step)
# mlflow.log_metric("val_loss", val_loss, step=step)`,
        pitfall: "Evaluating only training loss! A small model can memorize a tiny dataset to near-zero training loss while producing pure nonsense on new prompts.",
        project: "Log train and validation loss curves with matplotlib/MLflow on your TinyShakespeare run and identify the exact step where overfitting begins.",
        resourceTitle: "Andrej Karpathy — A Recipe for Training Neural Networks",
        resourceNote: "Timeless practical debugging checklist for diagnosing loss curves and silent bugs.",
        links: [
          { type: "Community Guide", label: "Andrej Karpathy — A Recipe for Training Neural Networks (Essential Blog)", url: "https://karpathy.github.io/2019/04/25/recipe/" },
          { type: "Community Guide", label: "Weights & Biases — Interpreting Train vs. Validation Loss Curves", url: "https://wandb.ai/site/articles/intro-to-loss-functions/" },
          { type: "Blog / Docs", label: "MLflow Official Docs — Tracking PyTorch Experiments", url: "https://mlflow.org/docs/latest/deep-learning/pytorch/index.html" }
        ]
      },
      {
        id: "p5-t5",
        codeNum: "5.5",
        title: "Autoregressive Sampling Strategies: Greedy, Temperature, Top-k & Top-p (Nucleus)",
        duration: "3 Days",
        plainEnglish: "'Autoregressive' just means predicting one word, gluing it onto the prompt, and feeding the whole thing back in to predict the next word. Always picking #1 ('Greedy') gets stuck in boring loops; 'Temperature' adjusts creativity, while 'Top-k' and 'Top-p' block absurd tail words from ever being chosen.",
        liveExample: "Production coding assistants (like Copilot) use low temperature (~0.1–0.2) so syntax stays valid, whereas creative brainstorming chat modes use temperature=0.8 + top_p=0.95—running the exact nucleus masking code you write here.",
        jargonTerms: [
          { term: "Autoregressive", def: "Generating text one token at a time, feeding each output back as new input." },
          { term: "Temperature", def: "Divides logits before Softmax: low (0.2) = focused/factual; high (1.1) = creative." },
          { term: "Top-k / Top-p", def: "Keeps only the top k words or top p% probability mass before sampling." }
        ],
        why: "Greedy decoding repeats loops ('I don't know. I don't know.'), while unfiltered random sampling picks bizarre low-probability words. Temperature, Top-k, and Top-p strike the sweet spot.",
        concepts: [
          "Autoregressive loop: crop context to last max_len tokens → forward pass → take last step logits[:, -1, :] → sample next ID → append.",
          "Temperature scaling (logits / T): T < 1 sharpens confident peaks; T > 1 flattens distribution for creativity.",
          "Top-k (keep k highest logits) and Top-p Nucleus sampling (keep smallest set of tokens whose cumulative probability ≥ p)."
        ],
        code: `sorted_logits, sorted_idx = torch.sort(logits / temperature, descending=True)
cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
sorted_mask = cum_probs > top_p
sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
sorted_mask[..., 0] = False
sorted_logits[sorted_mask] = float('-inf')`,
        pitfall: "Forgetting to slice context to the last max_len tokens (idx[:, -cfg.context_length:]) inside the generation loop—once generated text exceeds context_length, positional embeddings crash!",
        project: "Implement a unified generate(idx, max_new_tokens, temperature, top_k, top_p) method and compare outputs across decoding settings.",
        resourceTitle: "Hugging Face Blog — How to Generate Text (Top-k & Top-p)",
        resourceNote: "Visualizes probability bar charts before and after Temperature scaling and Top-k filtering.",
        links: [
          { type: "Community Guide", label: "Hugging Face Blog — How to Generate Text: Greedy, Beam, Top-k & Top-p", url: "https://huggingface.co/blog/how-to-generate" },
          { type: "Community Guide", label: "Cohere LLM University — Controlling LLM Outputs (Temperature, Top-k, Top-p)", url: "https://cohere.com/llmu/text-generation" },
          { type: "Community Guide", label: "Sebastian Raschka — Chapter 5 Decoding Strategies Code", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05" }
        ]
      }
    ]
  },
  {
    id: "phase-6",
    number: "06",
    category: "training",
    accent: "teal",
    duration: "Week 11–12",
    title: "Fine-Tuning, Alignment & Local Deployment",
    subtitle: "Load open pretrained weights into your own PyTorch code, fine-tune for chat instructions with LoRA, and deploy behind FastAPI + Ollama.",
    plainBigPicture: "A base pretrained model is only an autocomplete engine—not yet a helpful Q&A assistant. In this final phase, you load open pretrained weights into your own code, teach the model to answer questions using 'SFT' and 'LoRA' (memory-saving adapters), and wrap it in a live web API.",
    liveAiBridge: {
      headline: "Where Live Running AIs Use Phase 6 Skills",
      summary: "This is how real companies turn raw open-source base models (like Llama 3 or Qwen) into domain-specific enterprise chatbots, coding assistants, and private on-premise APIs.",
      examples: [
        "Enterprise Custom AI Assistants: Companies fine-tune open models on internal IT docs and support tickets using SFT (Module 6.2) + LoRA adapters (Module 6.3) on a single GPU.",
        "Running AI Offline on Laptops (Ollama & LM Studio): Powered by the 4-bit GGUF quantization concepts you learn in Module 6.3.",
        "Helpful & Safe Chatbot Behavior: Achieved via Direct Preference Optimization (DPO, Module 6.4) and served behind FastAPI microservices (Module 6.5)."
      ]
    },
    jargonDecoder: [
      { term: "SFT", def: "Fine-tuning a base model on Q&A chat pairs" },
      { term: "LoRA", def: "Training tiny adapter matrices instead of all weights" },
      { term: "Quantization", def: "Compressing weights (32-bit → 4-bit) to fit laptop RAM" },
      { term: "DPO / RLHF", def: "Aligning replies to human helpfulness preferences" },
      { term: "FastAPI / Ollama", def: "Serving your model locally as a REST API" }
    ],
    achievements: [
      "Map open pretrained weights (such as GPT-2 124M) directly into your from-scratch PyTorch classes and verify exact logit parity.",
      "Fine-tune a base LLM into a Q&A assistant using Supervised Fine-Tuning (SFT) and memory-saving LoRA adapters.",
      "Deploy your custom model locally behind a FastAPI + Uvicorn REST endpoint and benchmark it alongside Ollama."
    ],
    miniProject: "Fine-Tune a Small Model on Custom Q&A Data & Serve Locally via FastAPI + Uvicorn",
    topics: [
      {
        id: "p6-t1",
        codeNum: "6.1",
        title: "Loading Open Pretrained Weights (GPT-2 / Llama) into Custom PyTorch Code",
        duration: "3 Days",
        plainEnglish: "Training a smart model from scratch requires thousands of GPUs. Instead, companies release their trained weight files ('Checkpoints') for free! Here you copy those exact numbers into your own from-scratch PyTorch classes—proving your architecture is 100% authentic.",
        liveExample: "High-speed inference engines like vLLM, llama.cpp, and TensorRT-LLM rewrite the Transformer forward pass from scratch in C++/CUDA and map open Hugging Face safetensors checkpoints into their custom layers using this exact state_dict() matching process.",
        jargonTerms: [
          { term: "state_dict()", def: "Python dictionary mapping every layer name to its saved weight tensor." },
          { term: "Weight Mapping", def: "Copying pretrained tensors into matching layer names in your custom class." },
          { term: "Logit Parity", def: "Confirming your custom code outputs the exact same scores as the official model." }
        ],
        why: "Pretraining from scratch on web-scale data costs millions, so real-world builders map pretrained Hugging Face / OpenAI checkpoint tensors directly into their own clean PyTorch architecture.",
        concepts: [
          "Inspecting state_dict() keys and tensor shapes between your GPTModel and pretrained checkpoints.",
          "Handling Conv1D vs. nn.Linear weight transpositions when loading GPT-2 checkpoints.",
          "Verifying exact logit parity between your from-scratch code and the reference model."
        ],
        code: `my_sd = model.state_dict()
for k, v in list(my_sd.items())[:6]:
    print(f"{k:35s} | {tuple(v.shape)}")`,
        pitfall: "OpenAI's original GPT-2 checkpoints store linear weights transposed (in_features, out_features) compared to PyTorch's nn.Linear (out_features, in_features)—always check .shape and transpose where required!",
        project: "Load pretrained GPT-2 Small (124M) weights into your from-scratch PyTorch GPTModel and verify it completes prompts coherently out of the box.",
        resourceTitle: "Sebastian Raschka — Loading OpenAI Pretrained Weights & nanoGPT",
        resourceNote: "Step-by-step weight mapping function from pretrained tensors into custom GPTModel layers.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Chapter 5.5 Loading GPT-2 Weights Notebook", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05" },
          { type: "Community Guide", label: "Andrej Karpathy — nanoGPT from_pretrained GPT-2 Implementation", url: "https://github.com/karpathy/nanoGPT/blob/master/model.py" },
          { type: "YouTube", label: "Andrej Karpathy — Zero to Hero Playlist (GPT-2 Reproduction)", url: "https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ" }
        ]
      },
      {
        id: "p6-t2",
        codeNum: "6.2",
        title: "Supervised Fine-Tuning (SFT) & Instruction Dataset Formatting",
        duration: "3 Days",
        plainEnglish: "A raw pretrained model just mimics internet text—if you ask 'Explain DNS', it might reply 'Explain HTTP' because it thinks you're writing a list of exam questions! 'Supervised Fine-Tuning (SFT)' trains it on curated Instruction → Answer pairs so it behaves like a helpful assistant.",
        liveExample: "How Meta turned 'Llama-3-Base' into 'Llama-3-Instruct': they trained on millions of multi-turn ChatML conversations while masking user prompt tokens with ignore_index=-100 so the model only learns to generate helpful assistant replies.",
        jargonTerms: [
          { term: "SFT", def: "Supervised Fine-Tuning on formatted (Instruction, Response) examples." },
          { term: "Prompt Template", def: "Consistent wrapper tags like <|user|> ... <|assistant|> around chats." },
          { term: "ignore_index=-100", def: "Tells PyTorch's loss function to skip padding tokens when grading." }
        ],
        why: "A base LLM simply continues text (asking 'What is DNS?' might make it output 'What is DHCP?'). SFT teaches it to follow instructions and answer as an assistant.",
        concepts: [
          "Structuring instruction-input-response JSON records into consistent prompt templates (Alpaca / ChatML).",
          "Target padding with ignore_index = -100 so F.cross_entropy skips padding tokens and optionally masks user prompt tokens.",
          "Using a low learning rate (e.g., 2e-5 to 5e-5) for 2–3 epochs so the model adapts without forgetting base language skills."
        ],
        code: `loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)`,
        pitfall: "Training SFT for 20+ epochs on a tiny 500-example dataset—the model will memorize exact answers and lose general conversational ability (catastrophic forgetting).",
        project: "Curate a 100-item IT/Python Q&A dataset, format it with ChatML/Alpaca prompts, and fine-tune your model to answer domain questions.",
        resourceTitle: "Sebastian Raschka — Chapter 7: Instruction Fine-Tuning & Hugging Face SFT",
        resourceNote: "Covers custom collate functions, -100 padding masks, and instruction tuning end to end.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Chapter 7 Instruction Fine-Tuning Code", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07" },
          { type: "Community Guide", label: "Maxime Labonne — Fine-Tune Llama with Supervised Fine-Tuning (SFT) Guide", url: "https://mlabonne.github.io/blog/" },
          { type: "Blog / Docs", label: "Hugging Face LLM Course — Chapter 3: Fine-Tuning a Pretrained Model", url: "https://huggingface.co/learn/llm-course/chapter3/1" }
        ]
      },
      {
        id: "p6-t3",
        codeNum: "6.3",
        title: "Parameter-Efficient Fine-Tuning (LoRA) & Quantization Basics",
        duration: "3 Days",
        plainEnglish: "Updating billions of weights takes expensive server GPUs. 'LoRA' freezes the original brain and attaches tiny side-notebooks (matrices A and B) that learn only the new task—cutting memory by 90%! 'Quantization' rounds 32-bit decimals down to 4-bit integers so big models run smoothly on a regular laptop.",
        liveExample: "Cloud platforms like Together AI, Fireworks, and AWS Bedrock host hundreds of customer-specific LoRA adapters on top of a single frozen Llama base model in GPU RAM—swapping tiny ~50 MB A@B matrices per user request in milliseconds.",
        jargonTerms: [
          { term: "LoRA", def: "Low-Rank Adaptation—training tiny side matrices (A @ B) while base weights stay frozen." },
          { term: "Rank (r)", def: "Width of the tiny LoRA bottleneck (e.g., r=8 vs full width 4096)." },
          { term: "Quantization", def: "Compressing model weights from 32-bit floats to 8-bit or 4-bit numbers." },
          { term: "GGUF", def: "Popular compressed single-file format used by Ollama & llama.cpp to run LLMs locally." }
        ],
        why: "Full fine-tuning updates 100% of weights and needs huge VRAM. LoRA freezes base weights and trains tiny low-rank adapter matrices (~1% of parameters), while quantization shrinks RAM 4x.",
        concepts: [
          "LoRA math: freezing pretrained W (d×d) and learning ΔW = A @ B where A is (d×r) and B is (r×d) with rank r = 8 or 16.",
          "Initializing B = 0 and A ~ N(0, σ²) so ΔW = 0 at step 0—starting cleanly from the exact pretrained behavior.",
          "Post-training quantization intuition: FP32 (4 bytes) → BF16 (2 bytes) → INT8 (1 byte) → INT4/GGUF (0.5 bytes)."
        ],
        code: `class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scale = alpha / rank
    def forward(self, x):
        return (x @ self.A @ self.B) * self.scale`,
        pitfall: "Forgetting to freeze base model parameters (param.requires_grad = False) before training LoRA—otherwise PyTorch still computes and stores full gradients for the entire base model!",
        project: "Implement a LinearWithLoRA wrapper in pure PyTorch, attach it to your GPT's QKV projections, and verify trainable parameter count drops by >95%.",
        resourceTitle: "Sebastian Raschka — LoRA From Scratch in PyTorch (Appendix E)",
        resourceNote: "Implements LoRA from scratch in pure PyTorch with clear rank/alpha explanations.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Appendix E: LoRA From Scratch in PyTorch", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-E" },
          { type: "Community Guide", label: "Maarten Grootendorst — A Visual Guide to Quantization (GGUF, AWQ, GPTQ)", url: "https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization" },
          { type: "Blog / Docs", label: "Hugging Face PEFT Docs — LoRA Conceptual Guide", url: "https://huggingface.co/docs/peft/conceptual_guides/lora" }
        ]
      },
      {
        id: "p6-t4",
        codeNum: "6.4",
        title: "Evaluation Benchmarks, Hallucination Checks & Alignment (DPO Intuition)",
        duration: "2 Days",
        plainEnglish: "LLMs sometimes invent plausible-sounding nonsense ('Hallucinations'). Before shipping a model, we run automated 'pytest' checks on unseen prompts, set temperature=0 for repeatable grading, and use preference alignment ('DPO') to reward honest, helpful replies over unhelpful ones.",
        liveExample: "Before releasing Llama 3, Qwen, or Mistral, labs run Direct Preference Optimization (DPO) on human-ranked (chosen vs. rejected) pairs to curb hallucinations and run automated pytest-style eval harnesses across coding, math, and JSON formatting.",
        jargonTerms: [
          { term: "Hallucination", def: "When an LLM confidently generates false facts not supported by context." },
          { term: "DPO / RLHF", def: "Training the model to prefer human-approved answers over rejected ones." },
          { term: "pytest", def: "Python testing framework from your setup list for automated quality checks." }
        ],
        why: "Unlike unit tests, LLM outputs are open-ended. Evaluating quality requires automated benchmarks, deterministic rubric checks, and preference alignment (RLHF / DPO).",
        concepts: [
          "Automated evaluation: held-out task accuracy, exact-match/JSON schema validation, and LLM-as-a-Judge rubrics.",
          "Why base + SFT models still hallucinate or give unsafe answers—and how Direct Preference Optimization (DPO) trains on (chosen, rejected) response pairs.",
          "Deterministic inference (temperature=0.0) for reproducible evaluation runs."
        ],
        code: `@torch.no_grad()
def evaluate_prompt(model, prompt_ids):
    model.eval()
    return generate(model, prompt_ids, max_new_tokens=64, temperature=0.0)`,
        pitfall: "Evaluating your model on prompts that accidentally leaked into your SFT training split—always hold out a strict 10–15% test set of unseen instructions.",
        project: "Build a 20-prompt automated evaluation script using pytest that checks your fine-tuned model's outputs against expected keywords and format rules.",
        resourceTitle: "Sebastian Raschka — DPO Alignment & LLM Evaluation",
        resourceNote: "Demonstrates automated scoring of fine-tuned instruction responses.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Chapter 7.7 Evaluating Fine-Tuned LLMs & DPO", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07" },
          { type: "Community Guide", label: "Sebastian Raschka — LLM Alignment: RLHF vs. DPO Explained", url: "https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives" },
          { type: "Blog / Docs", label: "Hugging Face TRL Docs — Direct Preference Optimization (DPO)", url: "https://huggingface.co/docs/trl/main/en/dpo_trainer" }
        ]
      },
      {
        id: "p6-t5",
        codeNum: "6.5",
        title: "Serving Your Model Locally with FastAPI, Uvicorn & Ollama",
        duration: "3 Days",
        plainEnglish: "Finally, turn your Python script into a real web service! 'FastAPI' creates a clean HTTP endpoint (/v1/generate) where any app can send a prompt and get JSON back, 'Uvicorn' runs that web server locally, and 'Ollama' lets you run compressed open-source models locally for comparison.",
        liveExample: "Internal enterprise AI gateways and OpenAI-compatible local servers (like vLLM and Ollama's REST API) expose the exact POST /v1/generate JSON schema with Pydantic validation and Uvicorn workers that you build in this final module.",
        jargonTerms: [
          { term: "FastAPI", def: "Modern Python framework for building fast, auto-documented REST APIs." },
          { term: "Pydantic", def: "Data-validation library used by FastAPI to check incoming JSON types automatically." },
          { term: "Uvicorn", def: "Lightweight web server that runs your FastAPI app on localhost." },
          { term: "Ollama", def: "Local engine from your setup list for running quantized LLMs on your PC." }
        ],
        why: "Completing the engineering loop: package your PyTorch model behind a typed REST API using FastAPI + Uvicorn, and benchmark it alongside local GGUF models in Ollama.",
        concepts: [
          "Loading model weights once at FastAPI startup and running inference inside torch.no_grad().",
          "Validating request/response JSON payloads with Pydantic BaseModel.",
          "Running local quantized models via Ollama ('ollama run llama3.2:1b') and comparing latency/tokens-per-second."
        ],
        code: `from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI(title="My From-Scratch LLM API")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 60
    temperature: float = 0.7

@app.post("/v1/generate")
def generate_endpoint(req: GenerateRequest):
    with torch.no_grad():
        text = run_custom_gpt(req.prompt, req.max_tokens, req.temperature)
    return {"model": "mini-gpt-scratch", "output": text}`,
        pitfall: "Reloading the PyTorch checkpoint file inside the /generate route handler on every request—load your model once at module startup so API responses take milliseconds instead of seconds!",
        project: "Launch your custom LLM server with 'uvicorn app:app --reload', test it via Python requests or Swagger UI (/docs), and compare its output speed against Ollama.",
        resourceTitle: "FastAPI Official Docs & Ollama Quickstart",
        resourceNote: "Brings together PyTorch, FastAPI, Uvicorn, requests, and Ollama from your environment.",
        links: [
          { type: "Community Guide", label: "Real Python — Using FastAPI to Build Python Web APIs", url: "https://realpython.com/fastapi-python-web-apis/" },
          { type: "Blog / Docs", label: "FastAPI Official Tutorial — First Steps & Automatic /docs UI", url: "https://fastapi.tiangolo.com/tutorial/first-steps/" },
          { type: "Blog / Docs", label: "Ollama Official Quickstart & REST API Documentation", url: "https://docs.ollama.com/quickstart" }
        ]
      }
    ]
  }
];

const ALL_TOPICS = ROADMAP_PHASES.flatMap((phase) =>
  phase.topics.map((t) => ({
    ...t,
    phaseId: phase.id,
    phaseNumber: phase.number,
    phaseTitle: phase.title
  }))
);

const STORAGE_KEY = "llm_scratch_roadmap_completed_v2";

function getAccentClasses(accent) {
  const map = {
    teal: {
      badgeBg: "bg-teal-50 text-teal-800 border-teal-200",
      nodeRing: "border-teal-500 text-teal-700 bg-white",
      bar: "bg-teal-500",
      pillHover: "hover:border-teal-300"
    },
    indigo: {
      badgeBg: "bg-indigo-50 text-indigo-800 border-indigo-200",
      nodeRing: "border-indigo-500 text-indigo-700 bg-white",
      bar: "bg-indigo-500",
      pillHover: "hover:border-indigo-300"
    },
    amber: {
      badgeBg: "bg-amber-50 text-amber-800 border-amber-200",
      nodeRing: "border-amber-500 text-amber-700 bg-white",
      bar: "bg-amber-500",
      pillHover: "hover:border-amber-300"
    }
  };
  return map[accent] || map.teal;
}

// =========================================================================
// 2. MAIN REACT APPLICATION COMPONENT (COPY DIRECTLY INTO src/App.jsx)
// =========================================================================
export default function App() {
  const [completedIds, setCompletedIds] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      const parsed = saved ? JSON.parse(saved) : [];
      return new Set(Array.isArray(parsed) ? parsed : []);
    } catch {
      return new Set();
    }
  });

  const [currentFilter, setCurrentFilter] = useState("all");
  const [collapsedPhases, setCollapsedPhases] = useState(() => new Set());
  const [activeStageId, setActiveStageId] = useState(1);
  const [activeTopicId, setActiveTopicId] = useState(null);
  const [scrollSpinePct, setScrollSpinePct] = useState(0);
  const [toast, setToast] = useState(null);
  const [copiedSetup, setCopiedSetup] = useState(false);
  const timelineRef = useRef(null);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify([...completedIds]));
    } catch {
      // Ignore storage quota errors
    }
  }, [completedIds]);

  useEffect(() => {
    const handleScroll = () => {
      const el = timelineRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const windowH = window.innerHeight;
      const totalScrollable = rect.height - windowH * 0.4;
      if (totalScrollable <= 0) {
        setScrollSpinePct(100);
        return;
      }
      const scrolled = Math.min(Math.max(windowH * 0.5 - rect.top, 0), totalScrollable);
      setScrollSpinePct(Math.round((scrolled / totalScrollable) * 100));
    };
    handleScroll();
    window.addEventListener("scroll", handleScroll, { passive: true });
    window.addEventListener("resize", handleScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", handleScroll);
      window.removeEventListener("resize", handleScroll);
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === "Escape" && activeTopicId) {
        setActiveTopicId(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeTopicId]);

  const totalTopics = ALL_TOPICS.length;
  const doneCount = completedIds.size;
  const masteryPct = Math.round((doneCount / totalTopics) * 100);

  const activeStage = PIPELINE_STAGES.find((s) => s.id === activeStageId) || PIPELINE_STAGES[0];
  const activeTopicIndex = useMemo(
    () => ALL_TOPICS.findIndex((t) => t.id === activeTopicId),
    [activeTopicId]
  );
  const activeTopic = activeTopicIndex >= 0 ? ALL_TOPICS[activeTopicIndex] : null;

  const filteredPhases = useMemo(() => {
    if (currentFilter === "all") return ROADMAP_PHASES;
    return ROADMAP_PHASES.filter((p) => p.id === currentFilter);
  }, [currentFilter]);

  const toggleTopic = (topicId) => {
    setCompletedIds((prev) => {
      const next = new Set(prev);
      if (next.has(topicId)) next.delete(topicId);
      else next.add(topicId);
      return next;
    });
  };

  const togglePhaseCollapse = (phaseId) => {
    setCollapsedPhases((prev) => {
      const next = new Set(prev);
      if (next.has(phaseId)) next.delete(phaseId);
      else next.add(phaseId);
      return next;
    });
  };

  const handleJumpToPhase = (phaseId) => {
    if (currentFilter !== "all") setCurrentFilter("all");
    setCollapsedPhases((prev) => {
      const next = new Set(prev);
      next.delete(phaseId);
      return next;
    });
    setTimeout(() => {
      const target = document.getElementById(phaseId);
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 60);
  };

  return (
    <div className="min-h-screen bg-[#F4F7FA] text-[#17394D] font-sans antialiased selection:bg-teal-100 selection:text-teal-900">
      {/* Sticky Top Header */}
      <header className="sticky top-0 z-30 bg-white/90 backdrop-blur-md border-b border-slate-200/80 transition-all">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-teal-600 to-indigo-600 flex items-center justify-center text-white shadow-sm">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <circle cx="6" cy="6" r="2.5" />
                <circle cx="18" cy="6" r="2.5" />
                <circle cx="12" cy="18" r="2.5" />
                <path strokeLinecap="round" d="M8.2 7.2L10.8 15.8M15.8 7.2L13.2 15.8M8.5 6h7" />
              </svg>
            </div>
            <div>
              <span className="font-bold text-[#17394D] tracking-tight text-sm sm:text-base block leading-none">
                LLM From Scratch
              </span>
              <span className="text-[11px] font-medium text-slate-500">IT Beginner &rarr; PyTorch Builder</span>
            </div>
          </div>

          <div className="flex items-center gap-3 sm:gap-4 flex-1 max-w-md justify-end">
            <div className="flex-1 hidden sm:block">
              <div className="flex justify-between text-xs font-semibold mb-1">
                <span className="text-slate-600">Roadmap Mastery</span>
                <span className="text-teal-700 font-mono">
                  {masteryPct}% ({doneCount}/{totalTopics})
                </span>
              </div>
              <div className="w-full h-2 bg-slate-200/80 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-teal-500 via-indigo-500 to-amber-500 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${masteryPct}%` }}
                />
              </div>
            </div>

            <span className="sm:hidden text-xs font-mono font-semibold px-2.5 py-1 rounded-full bg-teal-50 text-teal-700 border border-teal-200">
              {masteryPct}%
            </span>

            <button
              type="button"
              onClick={() => setToast({ message: "Reset all saved roadmap checkmarks?", confirm: true })}
              title="Reset saved progress"
              className="text-xs font-medium text-slate-500 hover:text-rose-600 px-2.5 py-1.5 rounded-lg hover:bg-rose-50 transition-colors"
            >
              Reset
            </button>
          </div>
        </div>
      </header>

      {/* Hero Section + Interactive Pipeline + Step 0 Software/Package Setup */}
      <section className="relative pt-10 pb-8 sm:pt-14 sm:pb-12 border-b border-slate-200/70">
        <div className="max-w-6xl mx-auto px-4 sm:px-6">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-teal-200/80 shadow-xs text-xs font-semibold text-teal-800 mb-4">
              <span className="w-2 h-2 rounded-full bg-teal-500 animate-ping" />
              <span>Zero-to-GPT Engineering Blueprint &bull; Python 3.11 &amp; PyTorch</span>
            </div>
            <h1 className="text-3xl sm:text-5xl font-extrabold text-[#17394D] tracking-tight leading-[1.15]">
              Build Large Language Models{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-600 to-indigo-600">
                From Scratch
              </span>
            </h1>
            <p className="mt-4 text-base sm:text-lg text-slate-600 leading-relaxed">
              Starting your IT career and want to understand how GPT actually works under the hood? Follow this 6-phase interactive roadmap—from basic Python &amp; matrix multiplication to coding your own Transformer, pretraining it, and running it locally.
            </p>
          </div>

          {/* Interactive Live LLM Architecture Pipeline Visualizer */}
          <div className="mt-8 bg-white rounded-2xl border border-slate-200/90 shadow-sm p-5 sm:p-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-slate-100">
              <div>
                <h2 className="text-sm font-bold uppercase tracking-wider text-[#17394D] flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-indigo-500" />
                  What You Will Build: End-to-End Decoder-Only LLM Pipeline
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Click any stage in the pipeline below to inspect how raw text becomes next-token predictions—and jump to its phase.
                </p>
              </div>
              <div className="flex items-center gap-2 text-xs font-mono text-slate-500 bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200/70">
                <span>Input:</span>
                <span className="text-indigo-600 font-semibold">&quot;The capital of France is&quot;</span>
                <span>&rarr;</span>
                <span className="text-teal-700 font-semibold bg-teal-50 px-1.5 py-0.5 rounded">&quot;Paris&quot;</span>
              </div>
            </div>

            <div className="mt-5 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              {PIPELINE_STAGES.map((stage, idx) => {
                const isSelected = stage.id === activeStageId;
                return (
                  <button
                    key={stage.id}
                    type="button"
                    onClick={() => setActiveStageId(stage.id)}
                    className={`group relative text-left p-3.5 rounded-xl border transition-all duration-200 cursor-pointer ${
                      isSelected
                        ? "bg-gradient-to-br from-teal-50/90 to-indigo-50/70 border-teal-500 shadow-xs ring-2 ring-teal-500/20"
                        : "bg-slate-50/70 hover:bg-white border-slate-200/90 hover:border-indigo-300"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span
                        className={`w-2.5 h-2.5 rounded-full ${
                          isSelected ? "bg-teal-500" : "bg-slate-300 group-hover:bg-indigo-400"
                        }`}
                      />
                      {idx < PIPELINE_STAGES.length - 1 ? (
                        <span className="hidden lg:inline text-xs font-mono text-slate-400">&rarr;</span>
                      ) : (
                        <span className="text-[10px] font-mono text-teal-700 font-bold">OUT</span>
                      )}
                    </div>
                    <div className="text-xs font-bold text-[#17394D] group-hover:text-indigo-700 transition-colors">
                      {stage.name}
                    </div>
                    <div className="text-[11px] font-mono text-slate-500 mt-0.5">{stage.sub}</div>
                  </button>
                );
              })}
            </div>

            <div className="mt-4 p-3.5 rounded-xl bg-gradient-to-r from-slate-50 to-indigo-50/40 border border-slate-200/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3 transition-all">
              <div className="flex items-start sm:items-center gap-3">
                <span className="shrink-0 px-2.5 py-1 rounded-md text-xs font-mono font-semibold bg-indigo-100 text-indigo-800">
                  {activeStage.badge}
                </span>
                <p className="text-xs sm:text-sm text-[#17394D] font-medium">{activeStage.desc}</p>
              </div>
              <button
                type="button"
                onClick={() => handleJumpToPhase(activeStage.phaseId)}
                className="shrink-0 text-xs font-semibold text-teal-700 hover:text-teal-900 bg-white hover:bg-teal-50 border border-teal-200 px-3 py-1.5 rounded-lg shadow-2xs transition-all self-start sm:self-auto"
              >
                Jump to Phase &darr;
              </button>
            </div>
          </div>

          {/* Step 0: Environment Setup — Softwares, Python Packages & Terminal Commands */}
          <div className="mt-6 bg-white rounded-2xl border border-slate-200/90 shadow-sm p-5 sm:p-6 space-y-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-100">
              <div>
                <h2 className="text-sm font-bold uppercase tracking-wider text-[#17394D] flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-teal-500" />
                  Step 0: Required Softwares, Python Packages &amp; Setup Commands
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Install only what you need for building and running LLMs in Python &amp; PyTorch (no Java/JDK/Maven needed).
                </p>
              </div>
              <a
                href="https://pytorch.org/get-started/locally/"
                target="_blank"
                rel="noopener noreferrer"
                className="shrink-0 inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-700 hover:text-indigo-900 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 px-3 py-1.5 rounded-lg transition-colors self-start sm:self-auto"
              >
                <span>PyTorch GPU/CPU Selector</span>
                <span>&#8599;</span>
              </a>
            </div>

            {/* 4 Core Softwares */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {SETUP_SOFTWARES.map((sw) => (
                <a
                  key={sw.name}
                  href={sw.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group p-3.5 rounded-xl bg-slate-50/70 hover:bg-white border border-slate-200/90 hover:border-teal-400 transition-all flex flex-col justify-between gap-2.5"
                >
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-bold text-[#17394D] group-hover:text-teal-700">{sw.name}</span>
                      <span className="text-[11px] font-bold text-teal-700 shrink-0">Download &#8599;</span>
                    </div>
                    <p className="text-[11px] font-medium text-slate-700 leading-snug">{sw.role}</p>
                    <div className="p-2 rounded-lg bg-violet-50/75 border border-violet-200/75 text-[10px] text-[#17394D] leading-snug">
                      <span className="font-bold text-violet-900">Where Used: </span>
                      {sw.whereUsed}
                    </div>
                    <div className="p-2 rounded-lg bg-emerald-50/70 border border-emerald-200/75 text-[10px] text-[#17394D] leading-snug">
                      <span className="font-bold text-emerald-900">What You Build: </span>
                      {sw.whatYouBuild}
                    </div>
                  </div>
                  <code className="block text-[10px] font-mono bg-white border border-slate-200/80 rounded px-2 py-1 text-slate-600 truncate">
                    {sw.winget}
                  </code>
                </a>
              ))}
            </div>

            {/* 14 Python Packages + Copyable Setup Commands */}
            <div className="space-y-4">
              <div className="rounded-xl bg-sky-50/60 border border-sky-200/80 p-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
                  <div>
                    <span className="text-[11px] font-extrabold uppercase tracking-wider text-sky-950 block">
                      14 Python Packages (PyPI) — Purpose, Live AI Usage &amp; What You Will Build
                    </span>
                    <span className="text-[11px] text-slate-600">
                      Click any package badge to open its official PyPI documentation page.
                    </span>
                  </div>
                  <a
                    href="https://pypi.org/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[11px] font-semibold text-sky-800 hover:underline shrink-0"
                  >
                    pypi.org &#8599;
                  </a>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
                  {SETUP_PACKAGES.map((pkg) => (
                    <a
                      key={pkg.name}
                      href={`https://pypi.org/project/${pkg.name}/`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="group bg-white hover:bg-teal-50/40 border border-sky-200/90 hover:border-teal-400 rounded-xl p-3 text-[#17394D] transition-all flex flex-col justify-between gap-1.5"
                    >
                      <div>
                        <div className="flex items-center justify-between gap-2">
                          <strong className="font-mono text-xs text-indigo-700 group-hover:text-teal-700">
                            {pkg.name}
                          </strong>
                          <span className="text-[10px] font-semibold text-slate-400 group-hover:text-teal-700">
                            PyPI &#8599;
                          </span>
                        </div>
                        <p className="text-[11px] font-semibold text-[#17394D] mt-0.5 leading-snug">{pkg.desc}</p>
                      </div>
                      <div className="space-y-1 pt-1.5 border-t border-slate-100 text-[10px] leading-snug">
                        <p className="text-slate-600">
                          <strong className="text-violet-900">Where used:</strong> {pkg.whereUsed}
                        </p>
                        <p className="text-slate-700">
                          <strong className="text-emerald-900">You build:</strong> {pkg.whatYouBuild}
                        </p>
                      </div>
                    </a>
                  ))}
                </div>
                <p className="text-[11px] text-slate-600 mt-3 pt-2.5 border-t border-sky-200/70">
                  Tip: Always activate <code className="font-mono bg-white px-1 rounded text-[#17394D]">.venv</code> first so <code className="font-mono bg-white px-1 rounded text-[#17394D]">pip</code> installs cleanly inside your project folder.
                </p>
              </div>

              <div className="rounded-xl bg-slate-900 border border-slate-800 p-4 flex flex-col justify-between">
                <div className="flex items-center justify-between gap-2 mb-2">
                  <span className="text-[11px] font-mono font-semibold text-teal-400">
                    Windows PowerShell Setup &amp; Verification Commands
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      navigator.clipboard?.writeText(SETUP_COMMANDS_TEXT);
                      setCopiedSetup(true);
                      setTimeout(() => setCopiedSetup(false), 2200);
                    }}
                    className="text-[11px] font-semibold px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
                  >
                    {copiedSetup ? "Copied!" : "Copy Commands"}
                  </button>
                </div>
                <pre className="text-[11px] font-mono text-slate-100 overflow-x-auto leading-relaxed">
                  <code>{SETUP_COMMANDS_TEXT}</code>
                </pre>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Filter Bar & Controls */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 pt-8 pb-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-2" role="tablist" aria-label="Roadmap Phase Filters">
            {[
              { id: "all", label: "All Phases (6)" },
              { id: "phase-1", label: "Phase 1: Foundations" },
              { id: "phase-2", label: "Phase 2: Autograd" },
              { id: "phase-3", label: "Phase 3: Tokenization" },
              { id: "phase-4", label: "Phase 4: Transformer" },
              { id: "phase-5", label: "Phase 5: Pretraining" },
              { id: "phase-6", label: "Phase 6: Fine-Tune & Serve" }
            ].map((tab) => {
              const isMatch = currentFilter === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setCurrentFilter(tab.id)}
                  className={`px-3 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all ${
                    isMatch
                      ? "bg-[#17394D] text-white shadow-xs"
                      : "bg-white text-slate-600 hover:text-[#17394D] border border-slate-200/90 hover:border-slate-300"
                  }`}
                >
                  {tab.label}
                </button>
              );
            })}
          </div>

          <div className="flex items-center gap-2 self-start md:self-auto">
            <button
              type="button"
              onClick={() => setCollapsedPhases(new Set())}
              className="text-xs font-semibold text-slate-600 hover:text-[#17394D] bg-white border border-slate-200 px-3 py-2 rounded-xl hover:bg-slate-50 transition"
            >
              Expand All
            </button>
            <button
              type="button"
              onClick={() => setCollapsedPhases(new Set(ROADMAP_PHASES.map((p) => p.id)))}
              className="text-xs font-semibold text-slate-600 hover:text-[#17394D] bg-white border border-slate-200 px-3 py-2 rounded-xl hover:bg-slate-50 transition"
            >
              Collapse All
            </button>
          </div>
        </div>
      </section>

      {/* Main Timeline Roadmap */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-6 pb-24">
        <div className="relative" ref={timelineRef}>
          <div className="hidden md:block absolute left-7 top-4 bottom-8 w-1 bg-slate-200/90 rounded-full overflow-hidden">
            <div
              className="w-full bg-gradient-to-b from-teal-500 via-indigo-500 to-amber-500 rounded-full transition-all duration-150 ease-out"
              style={{ height: `${scrollSpinePct}%` }}
            />
          </div>

          <div className="space-y-8">
            {filteredPhases.map((phase) => {
              const theme = getAccentClasses(phase.accent);
              const phaseDoneCount = phase.topics.filter((t) => completedIds.has(t.id)).length;
              const phaseTotal = phase.topics.length;
              const pct = Math.round((phaseDoneCount / phaseTotal) * 100);
              const isPhaseComplete = phaseDoneCount === phaseTotal;
              const isCollapsed = collapsedPhases.has(phase.id);

              return (
                <article key={phase.id} id={phase.id} className="relative md:pl-20">
                  <div
                    className={`hidden md:flex absolute left-0 top-5 w-14 h-14 rounded-2xl border-2 ${
                      isPhaseComplete ? "border-emerald-500 bg-emerald-500 text-white" : theme.nodeRing
                    } shadow-sm items-center justify-center font-mono font-extrabold text-base z-10 transition-colors`}
                  >
                    {isPhaseComplete ? "✓" : phase.number}
                  </div>

                  <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm hover:shadow-md transition-all duration-300 overflow-hidden">
                    <div
                      onClick={() => togglePhaseCollapse(phase.id)}
                      className="p-5 sm:p-6 cursor-pointer select-none flex flex-col sm:flex-row sm:items-center justify-between gap-4 hover:bg-slate-50/50 transition-colors"
                    >
                      <div className="space-y-1.5 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="md:hidden font-mono text-xs font-bold px-2 py-0.5 rounded-md bg-[#17394D] text-white">
                            Phase {phase.number}
                          </span>
                          <span className={`text-xs font-mono font-semibold px-2.5 py-0.5 rounded-full border ${theme.badgeBg}`}>
                            {phase.duration}
                          </span>
                          <span className="text-xs font-medium text-slate-400">
                            &bull; {phaseDoneCount}/{phaseTotal} modules completed
                          </span>
                        </div>
                        <h2 className="text-lg sm:text-xl font-extrabold text-[#17394D] tracking-tight">{phase.title}</h2>
                        <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">{phase.subtitle}</p>

                        <div className="mt-2.5 p-3 rounded-xl bg-sky-50/75 border border-sky-200/80 text-xs text-[#17394D] leading-relaxed">
                          <span className="font-bold text-sky-900 uppercase tracking-wider text-[10px] mr-1.5">
                            Plain-English Big Picture:
                          </span>
                          {phase.plainBigPicture}
                        </div>

                        {phase.liveAiBridge && (
                          <div className="mt-2.5 p-3.5 rounded-xl bg-gradient-to-r from-violet-50/85 via-indigo-50/60 to-white border border-violet-200/85 text-xs text-[#17394D] leading-relaxed">
                            <div className="flex items-center gap-1.5 mb-1">
                              <span className="w-2 h-2 rounded-full bg-violet-600 animate-pulse" />
                              <span className="font-extrabold text-violet-950 uppercase tracking-wider text-[10px]">
                                {phase.liveAiBridge.headline}
                              </span>
                            </div>
                            <p className="text-slate-700 mb-2">{phase.liveAiBridge.summary}</p>
                            <ul className="space-y-1">
                              {phase.liveAiBridge.examples.map((ex, idx) => (
                                <li key={idx} className="flex items-start gap-1.5 text-[11px] text-[#17394D]">
                                  <span className="text-violet-600 font-bold shrink-0 mt-px">&rarr;</span>
                                  <span>{ex}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        <div className="mt-2 flex flex-wrap items-center gap-1.5 pt-0.5">
                          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mr-1">
                            Jargon Decoder:
                          </span>
                          {phase.jargonDecoder.map((item) => (
                            <span
                              key={item.term}
                              className="inline-flex items-center gap-1 text-[11px] bg-slate-50 border border-slate-200/90 rounded-lg px-2 py-0.5 text-slate-700"
                            >
                              <strong className="font-mono text-indigo-700">{item.term}</strong>
                              <span className="text-slate-400">=</span>
                              <span>{item.def}</span>
                            </span>
                          ))}
                        </div>
                      </div>

                      <div className="flex items-center justify-between sm:justify-end gap-4 shrink-0 pt-2 sm:pt-0 border-t sm:border-t-0 border-slate-100">
                        <div className="w-28">
                          <div className="flex justify-between text-[11px] font-mono font-semibold text-slate-500 mb-1">
                            <span>Progress</span>
                            <span>{pct}%</span>
                          </div>
                          <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                            <div
                              className={`h-full ${isPhaseComplete ? "bg-emerald-500" : theme.bar} rounded-full transition-all duration-500`}
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                        </div>
                        <button
                          type="button"
                          aria-label="Toggle phase details"
                          className={`w-8 h-8 rounded-lg bg-slate-100 text-slate-600 flex items-center justify-center transition-transform duration-300 ${
                            isCollapsed ? "rotate-180" : ""
                          }`}
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2.2" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                          </svg>
                        </button>
                      </div>
                    </div>

                    {!isCollapsed && (
                      <div className="px-5 sm:px-6 pb-6 pt-2 border-t border-slate-100 space-y-4">
                        <div className="mt-2 rounded-xl bg-gradient-to-r from-indigo-50/80 via-sky-50/45 to-white border border-indigo-200/80 p-4">
                          <div className="flex items-center gap-2 mb-2.5">
                            <span className="text-[11px] font-extrabold uppercase tracking-wider text-indigo-950">
                              What You Will Learn in Phase {phase.number} ({phaseTotal} Modules)
                            </span>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {phase.topics.map((t) => (
                              <button
                                key={t.id}
                                type="button"
                                onClick={() => setActiveTopicId(t.id)}
                                className="inline-flex items-center gap-1.5 text-left bg-white/90 hover:bg-indigo-50/80 border border-indigo-200/75 hover:border-indigo-400 rounded-lg px-2.5 py-1.5 text-xs text-[#17394D] font-semibold transition-all shadow-2xs"
                              >
                                <span className="font-mono text-[11px] font-bold text-indigo-700">{t.codeNum}</span>
                                <span>{t.title}</span>
                              </button>
                            ))}
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                          {phase.topics.map((topic) => {
                            const checked = completedIds.has(topic.id);
                            return (
                              <div
                                key={topic.id}
                                onClick={() => setActiveTopicId(topic.id)}
                                className={`group relative flex items-start gap-3 p-4 rounded-xl border transition-all duration-200 cursor-pointer ${
                                  checked
                                    ? "bg-emerald-50/45 border-emerald-200/90"
                                    : `bg-slate-50/60 border-slate-200/80 ${theme.pillHover} hover:bg-white hover:shadow-xs`
                                }`}
                              >
                                <div className="pt-0.5 shrink-0" onClick={(e) => e.stopPropagation()}>
                                  <input
                                    type="checkbox"
                                    checked={checked}
                                    onChange={() => toggleTopic(topic.id)}
                                    aria-label={`Mark ${topic.title} completed`}
                                    className="w-4 h-4 rounded text-teal-600 focus:ring-teal-500 border-slate-300 cursor-pointer"
                                  />
                                </div>

                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center justify-between gap-2">
                                    <div className="flex items-center gap-1.5">
                                      <span className="text-[11px] font-mono font-bold px-1.5 py-0.5 rounded bg-white border border-slate-200/90 text-[#17394D]">
                                        {topic.codeNum}
                                      </span>
                                      <span className="text-[11px] font-mono text-slate-400">{topic.duration}</span>
                                    </div>
                                    <span className="text-[11px] font-semibold text-indigo-600 opacity-85 group-hover:opacity-100 group-hover:translate-x-0.5 transition-all shrink-0">
                                      Deep Dive &rarr;
                                    </span>
                                  </div>

                                  <h3
                                    className={`text-xs sm:text-sm font-bold leading-snug mt-1.5 ${
                                      checked
                                        ? "text-slate-500 line-through decoration-emerald-400"
                                        : "text-[#17394D] group-hover:text-indigo-950"
                                    }`}
                                  >
                                    {topic.title}
                                  </h3>

                                  <div
                                    className={`mt-2 p-2.5 rounded-lg bg-sky-50/70 border border-sky-200/70 text-[11px] leading-relaxed ${
                                      checked ? "text-slate-400" : "text-[#17394D]"
                                    }`}
                                  >
                                    <span className="font-bold text-sky-900">What is this? </span>
                                    {topic.plainEnglish}
                                  </div>

                                  {topic.liveExample && (
                                    <div
                                      className={`mt-2 p-2.5 rounded-lg bg-violet-50/75 border border-violet-200/80 text-[11px] leading-relaxed ${
                                        checked ? "text-slate-400" : "text-[#17394D]"
                                      }`}
                                    >
                                      <span className="font-bold text-violet-900">Live AI In Action: </span>
                                      {topic.liveExample}
                                    </div>
                                  )}

                                  <div className="mt-2 flex flex-wrap gap-1">
                                    {topic.jargonTerms.map((j) => (
                                      <span
                                        key={j.term}
                                        className="inline-block text-[10px] leading-snug bg-white border border-slate-200/85 rounded-md px-1.5 py-0.5 text-slate-600"
                                      >
                                        <strong className="font-mono text-sky-900">{j.term}:</strong> {j.def}
                                      </span>
                                    ))}
                                  </div>

                                  <ul className="mt-2.5 space-y-1 border-t border-slate-200/60 pt-2">
                                    {topic.concepts.map((concept, idx) => (
                                      <li
                                        key={idx}
                                        className={`flex items-start gap-1.5 text-[11px] leading-snug ${
                                          checked ? "text-slate-400" : "text-slate-600"
                                        }`}
                                      >
                                        <span className="text-teal-600 font-bold shrink-0 mt-px">&bull;</span>
                                        <span>{concept}</span>
                                      </li>
                                    ))}
                                  </ul>

                                  <div
                                    className="mt-2.5 pt-2 border-t border-slate-200/60 flex flex-wrap items-center gap-1.5"
                                    onClick={(e) => e.stopPropagation()}
                                  >
                                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mr-0.5">
                                      Learn:
                                    </span>
                                    {topic.links.map((link, idx) => (
                                      <a
                                        key={idx}
                                        href={link.url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        title={link.label}
                                        className={`inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-md border transition-colors ${
                                          link.type === "YouTube"
                                            ? "bg-rose-50/80 text-rose-700 border-rose-200/90 hover:bg-rose-100"
                                            : link.type === "Community Guide"
                                              ? "bg-teal-50/80 text-teal-800 border-teal-200/90 hover:bg-teal-100"
                                              : "bg-indigo-50/80 text-indigo-700 border-indigo-200/90 hover:bg-indigo-100"
                                        }`}
                                      >
                                        <span className="truncate max-w-[165px]">
                                          {link.label.split("—")[0].trim()}
                                        </span>
                                        <span>&#8599;</span>
                                      </a>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>

                        <div className="rounded-xl bg-gradient-to-r from-emerald-50/80 via-teal-50/45 to-white border border-emerald-200/85 p-4">
                          <div className="text-[11px] font-extrabold uppercase tracking-wider text-emerald-900 mb-2.5">
                            What You Will Achieve &amp; Master After Phase {phase.number}
                          </div>
                          <ul className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                            {phase.achievements.map((item, idx) => (
                              <li
                                key={idx}
                                className="flex items-start gap-2 bg-white/85 rounded-lg p-2.5 border border-emerald-200/60 text-xs text-[#17394D] leading-snug"
                              >
                                <span className="text-emerald-600 font-extrabold shrink-0 mt-px">&#10003;</span>
                                <span>{item}</span>
                              </li>
                            ))}
                          </ul>
                        </div>

                        <div className="rounded-xl bg-gradient-to-r from-amber-50/80 via-orange-50/40 to-white border border-amber-200/80 p-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                          <div>
                            <span className="text-[10px] font-bold uppercase tracking-wider text-amber-800 block">
                              Phase {phase.number} Milestone Build
                            </span>
                            <span className="text-xs sm:text-sm font-bold text-[#17394D]">{phase.miniProject}</span>
                          </div>
                          <button
                            type="button"
                            onClick={() => setActiveTopicId(phase.topics[phase.topics.length - 1].id)}
                            className="text-xs font-semibold text-amber-900 hover:text-amber-700 underline underline-offset-2 self-start sm:self-center shrink-0"
                          >
                            Inspect Build Details
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </article>
              );
            })}
          </div>
        </div>

        {/* Bottom Completion Banner */}
        <div className="mt-12 bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 sm:p-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div>
            <span className="inline-block text-xs font-bold uppercase tracking-wider text-teal-700 bg-teal-50 border border-teal-200 px-2.5 py-1 rounded-md mb-2">
              Golden Rule for Beginners
            </span>
            <h3 className="text-xl font-bold text-[#17394D]">
              Don&apos;t just watch tutorials—print tensor shapes at every step!
            </h3>
            <p className="text-sm text-slate-600 mt-1.5 max-w-2xl">
              90% of bugs when building LLMs from scratch come from mismatched matrix dimensions{" "}
              <code className="font-mono text-xs bg-slate-100 px-1.5 py-0.5 rounded text-indigo-700">
                (Batch, Seq_Len, Embed_Dim)
              </code>
              . Keep JupyterLab open beside Andrej Karpathy&apos;s or Sebastian Raschka&apos;s videos and code every single layer yourself.
            </p>
          </div>
          <div className="shrink-0 flex flex-col sm:flex-row gap-3 w-full md:w-auto">
            <button
              type="button"
              onClick={() => setActiveTopicId(ALL_TOPICS[0].id)}
              className="px-5 py-3 rounded-xl bg-gradient-to-r from-teal-600 to-indigo-600 hover:from-teal-700 hover:to-indigo-700 text-white font-semibold text-sm shadow-sm transition-all text-center"
            >
              Start Phase 1, Topic 1 &rarr;
            </button>
          </div>
        </div>
      </main>

      {/* Slide-Over Detail Drawer */}
      {activeTopic && (
        <>
          <div
            onClick={() => setActiveTopicId(null)}
            className="fixed inset-0 bg-[#17394D]/25 backdrop-blur-xs z-40 transition-opacity"
            aria-hidden="true"
          />
          <aside
            className="fixed top-0 right-0 h-full w-full max-w-xl bg-white z-50 shadow-2xl border-l border-slate-200 flex flex-col"
            aria-label="Topic Details"
          >
            <div className="px-6 py-5 border-b border-slate-100 flex items-start justify-between gap-4 bg-slate-50/70">
              <div>
                <div className="flex items-center gap-2 mb-1.5">
                  <span className="text-[11px] font-bold uppercase tracking-wider px-2.5 py-0.5 rounded-full bg-teal-100 text-teal-800">
                    Phase {activeTopic.phaseNumber} &bull; Module {activeTopic.codeNum}
                  </span>
                  <span className="text-xs font-mono text-slate-500">{activeTopic.duration}</span>
                </div>
                <h2 className="text-xl font-extrabold text-[#17394D] leading-snug">{activeTopic.title}</h2>
              </div>
              <button
                type="button"
                onClick={() => setActiveTopicId(null)}
                className="p-2 rounded-xl text-slate-400 hover:text-[#17394D] hover:bg-slate-200/60 transition-colors"
                aria-label="Close drawer"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2.2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="p-6 overflow-y-auto flex-1 space-y-6">
              <div className="bg-gradient-to-br from-sky-50/90 to-indigo-50/50 border border-sky-200/80 rounded-xl p-4">
                <h3 className="text-xs font-bold uppercase tracking-wider text-sky-900 mb-2">
                  Plain-English Analogy &amp; Jargon Breakdown
                </h3>
                <p className="text-sm text-[#17394D] leading-relaxed mb-3">{activeTopic.plainEnglish}</p>
                <div className="space-y-1.5 border-t border-sky-200/70 pt-2.5">
                  {activeTopic.jargonTerms.map((j) => (
                    <div key={j.term} className="flex items-start gap-2 text-xs text-[#17394D]">
                      <span className="font-mono font-bold text-sky-900 bg-white px-1.5 py-0.5 rounded border border-sky-200/90 shrink-0">
                        {j.term}
                      </span>
                      <span className="text-slate-700 pt-0.5 leading-snug">{j.def}</span>
                    </div>
                  ))}
                </div>
              </div>

              {activeTopic.liveExample && (
                <div className="bg-gradient-to-br from-violet-50/90 to-indigo-50/50 border border-violet-200/85 rounded-xl p-4">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-violet-900 mb-1.5 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-violet-600" />
                    Live Production AI In Action (Real-World Application)
                  </h3>
                  <p className="text-sm text-[#17394D] leading-relaxed">{activeTopic.liveExample}</p>
                </div>
              )}

              <div className="bg-gradient-to-br from-teal-50/80 to-emerald-50/40 border border-teal-200/70 rounded-xl p-4">
                <h3 className="text-xs font-bold uppercase tracking-wider text-teal-800 mb-1.5">
                  Why It Matters for Building LLMs
                </h3>
                <p className="text-sm text-[#17394D] leading-relaxed">{activeTopic.why}</p>
              </div>

              <div>
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-3">
                  Key Concepts to Master
                </h3>
                <ul className="space-y-2.5">
                  {activeTopic.concepts.map((c, idx) => (
                    <li key={idx} className="flex items-start gap-2.5 text-xs sm:text-sm text-[#17394D]">
                      <span className="w-5 h-5 rounded-full bg-teal-50 text-teal-700 border border-teal-200 flex items-center justify-center text-xs font-bold shrink-0 mt-0.5">
                        &#10003;
                      </span>
                      <span className="leading-relaxed">{c}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                    Mental Model / PyTorch Peek
                  </h3>
                  <span className="text-[11px] font-mono text-slate-400">Python 3.11</span>
                </div>
                <pre className="bg-slate-900 text-slate-100 rounded-xl p-4 text-xs font-mono overflow-x-auto leading-relaxed border border-slate-800">
                  <code>{activeTopic.code}</code>
                </pre>
              </div>

              <div className="bg-rose-50/75 border border-rose-200/80 rounded-xl p-4">
                <h3 className="text-xs font-bold uppercase tracking-wider text-rose-800 mb-1.5">
                  Beginner Pitfall &amp; Debugging Tip
                </h3>
                <p className="text-xs sm:text-sm text-[#17394D] leading-relaxed">{activeTopic.pitfall}</p>
              </div>

              <div className="bg-amber-50/70 border border-amber-200/80 rounded-xl p-4">
                <h3 className="text-xs font-bold uppercase tracking-wider text-amber-800 mb-1.5">
                  Hands-On Mini-Project
                </h3>
                <p className="text-sm text-[#17394D] font-medium leading-relaxed">{activeTopic.project}</p>
              </div>

              <div className="bg-indigo-50/70 border border-indigo-200/80 rounded-xl p-4">
                <h3 className="text-xs font-bold uppercase tracking-wider text-indigo-800 mb-1.5">
                  Curated Study Links (YouTube + Best Blogs &amp; Docs)
                </h3>
                <p className="text-sm font-bold text-[#17394D]">{activeTopic.resourceTitle}</p>
                <p className="text-xs text-slate-600 mt-1 mb-3">{activeTopic.resourceNote}</p>
                <div className="space-y-2">
                  {activeTopic.links.map((link, idx) => (
                    <a
                      key={idx}
                      href={link.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center justify-between gap-2.5 p-2.5 rounded-lg bg-white border border-indigo-200/80 hover:border-indigo-400 hover:shadow-2xs transition-all group"
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <span
                          className={`shrink-0 text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded ${
                            link.type === "YouTube"
                              ? "bg-rose-100 text-rose-800"
                              : link.type === "Community Guide"
                                ? "bg-teal-100 text-teal-800"
                                : "bg-indigo-100 text-indigo-800"
                          }`}
                        >
                          {link.type}
                        </span>
                        <span className="text-xs font-semibold text-[#17394D] group-hover:text-indigo-700 truncate">
                          {link.label}
                        </span>
                      </div>
                      <span className="text-xs font-bold text-indigo-600 shrink-0 group-hover:translate-x-0.5 transition-transform">
                        Open &#8599;
                      </span>
                    </a>
                  ))}
                </div>
              </div>
            </div>

            <div className="px-6 py-4 border-t border-slate-200 bg-slate-50 flex items-center justify-between gap-4">
              <button
                type="button"
                disabled={activeTopicIndex <= 0}
                onClick={() => activeTopicIndex > 0 && setActiveTopicId(ALL_TOPICS[activeTopicIndex - 1].id)}
                className="px-3.5 py-2 rounded-xl border border-slate-200 bg-white text-xs font-semibold text-slate-600 hover:text-[#17394D] hover:bg-slate-100 transition disabled:opacity-40"
              >
                &larr; Prev Topic
              </button>

              <label className="flex items-center gap-2.5 cursor-pointer select-none px-4 py-2 rounded-xl bg-white border border-slate-200 hover:border-teal-400 transition shadow-2xs">
                <input
                  type="checkbox"
                  checked={completedIds.has(activeTopic.id)}
                  onChange={() => toggleTopic(activeTopic.id)}
                  className="w-4 h-4 rounded text-teal-600 focus:ring-teal-500 border-slate-300 cursor-pointer"
                />
                <span className="text-xs sm:text-sm font-bold text-[#17394D]">Mark Topic Completed</span>
              </label>

              <button
                type="button"
                disabled={activeTopicIndex >= ALL_TOPICS.length - 1}
                onClick={() =>
                  activeTopicIndex < ALL_TOPICS.length - 1 && setActiveTopicId(ALL_TOPICS[activeTopicIndex + 1].id)
                }
                className="px-3.5 py-2 rounded-xl border border-slate-200 bg-white text-xs font-semibold text-slate-600 hover:text-[#17394D] hover:bg-slate-100 transition disabled:opacity-40"
              >
                Next Topic &rarr;
              </button>
            </div>
          </aside>
        </>
      )}

      {/* Non-Blocking Toast Banner */}
      {toast && (
        <div className="fixed bottom-5 right-5 z-50 bg-[#17394D] text-white px-4 py-3 rounded-xl shadow-lg flex items-center gap-3 text-xs sm:text-sm">
          <span>{toast.message}</span>
          {toast.confirm && (
            <button
              type="button"
              onClick={() => {
                setCompletedIds(new Set());
                setToast({ message: "Roadmap progress cleared.", confirm: false });
                setTimeout(() => setToast(null), 2500);
              }}
              className="px-2.5 py-1 rounded-lg bg-rose-500 hover:bg-rose-600 text-white font-semibold text-xs"
            >
              Yes, Reset
            </button>
          )}
          <button
            type="button"
            onClick={() => setToast(null)}
            className="text-slate-300 hover:text-white font-semibold text-xs"
          >
            Dismiss
          </button>
        </div>
      )}
    </div>
  );
}