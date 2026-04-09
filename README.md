# 🛡️ AI-Driven Game Theoretic Cybersecurity Defense Simulator

An interactive simulation platform that models the attacker-defender interaction in a computer network using **Game Theory** and **Adaptive AI**.

---

## 🎯 Features

| Feature | Details |
|---|---|
| **Game-Theoretic Engine** | Nash Equilibria (pure & mixed), Pareto Optimality via `nashpy` |
| **Network Visualization** | Interactive drag-and-drop topology using `PyVis` |
| **Repeated Simulation** | 20–500 rounds with strategy evolution plots |
| **Q-Learning AI** | Tabular RL defender agent that learns from attacker patterns |
| **Pattern Recognition** | `RandomForestClassifier` predicts next attacker move |
| **LP Comparison** | `scipy.optimize.linprog` vs Nash equilibrium efficiency |
| **CSV Export** | Download all results and configurations |

---

## 📁 Project Structure

```
PROJECT_TECH_OPT/
├── app.py                    ← Main Streamlit app (run this)
├── requirements.txt          ← All dependencies
│
├── engine/
│   ├── game_theory.py        ← Nash equilibria + Pareto (nashpy)
│   ├── network_model.py      ← NetworkX graph topology
│   ├── simulation.py         ← Repeated game engine
│   └── optimization.py       ← SciPy LP comparison
│
├── ai/
│   ├── q_learning.py         ← Q-learning defender agent
│   └── pattern_recognition.py← sklearn attack classifier
│
├── ui/
│   ├── network_viz.py        ← PyVis interactive graph renderer
│   ├── charts.py             ← Plotly charts
│   └── matrix_editor.py      ← Streamlit matrix editor widgets
│
└── utils/
    └── export.py             ← CSV export utilities
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 🧭 Dashboard Guide

### Sidebar
- **Scenario Preset** — Choose a pre-built game or create a custom one
- **Strategy Configuration** — Set number of strategies and names
- **Payoff Matrices** — Edit defender (A) and attacker (B) matrices directly
- **Simulation Parameters** — Rounds, AI toggle, best-response dynamics
- **▶ Run Full Analysis** — Triggers full computation pipeline

### Tabs

| Tab | Purpose |
|-----|---------|
| 🌐 **Network** | View & interact with the network topology |
| 🎯 **Game Analysis** | Nash equilibria, payoff heatmaps, Pareto frontier |
| 🔄 **Simulation** | Strategy evolution and payoff convergence plots |
| 🤖 **AI Insights** | Q-table, learned policy, attack pattern predictions |
| ⚖️ **Comparison** | Nash Equilibrium vs. LP Optimal payoff comparison |
| 💾 **Export** | Download results as CSV |

---

## 🧩 Tech Stack

| Component | Library |
|-----------|---------|
| Frontend | Streamlit |
| Game Theory | nashpy |
| Graph Topology | NetworkX |
| Interactive Network | PyVis |
| Charts | Plotly |
| AI / RL | Custom Q-Learning |
| Pattern Recognition | scikit-learn (RandomForest) |
| LP Optimization | SciPy linprog |
| Data | pandas, numpy |

---

## 📊 Example Scenarios

### Zero-Sum Game
- Defender: Patch System / Monitor Network  
- Attacker: Exploit Vulnerability / Phishing  
- Classic competitive scenario with pure Nash Equilibrium

### General-Sum (Realistic)
- Both players have independent costs and benefits
- Mixed strategy equilibrium reflects realistic cyber deterrence

### APT (Advanced Persistent Threat)
- 4×4 matrix with attacker patience built in
- Shows how asymmetric cybersecurity games evolve

---

## 🤖 AI Components Explained

### Q-Learning Defender
- **State**: Last observed attacker strategy
- **Action**: Defender strategy to deploy
- **Reward**: Defender's payoff from the game matrix
- **Update**: Bellman equation with ε-greedy exploration decay
- **Result**: Optimal defensive policy emerges over repeated play

### Attack Pattern Classifier (RandomForest)
- **Features**: Sliding window of last 3 attacker moves
- **Target**: Predicted next attacker strategy
- **Training**: 300 synthetic + simulation history samples
- **Output**: Probability distribution over all attacker strategies

---

## ⚖️ Nash vs LP Explained

- **Nash Equilibrium**: Decentralized stable outcome — no player benefits from deviating unilaterally
- **LP Optimal**: Minimax strategy solved via linear programming (centralized)
- For **zero-sum games**: both methods coincide (minimax theorem)
- For **general-sum games**: efficiency gap may exist — shown in the comparison tab

---

## 📄 License

MIT License — free for academic, research, and educational use.
