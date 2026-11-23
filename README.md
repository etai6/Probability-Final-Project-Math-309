# How Long Until Random Is Random?
An interactive explorer of **mixing times** and the **cutoff phenomenon**, built with Streamlit and Plotly.

## What it shows
- **Hypercube walk** – a row of *d* light switches re-set one at a time. Exact distance-to-random curve,
  the distribution a "detective" sees, and a multi-size comparison where the cliff sharpens.
- **Card shuffling** – exact Bayer–Diaconis distance for riffle shuffles of an *n*-card deck
  (7 shuffles gives distance ≈ 0.33 for 52 cards), plus a simulated heat-strip of a deck being shuffled.
- **Plain-language tab** and **Math notes** tab (definitions, theorems, references).

All distance curves are computed **exactly** (validated against brute-force enumeration); only the
heat-strip uses random simulation.

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free
Push this folder to GitHub, then create an app at https://share.streamlit.io pointing to `app.py`.

## References
Aldous & Diaconis (1986); Bayer & Diaconis (1992); Diaconis (1996); Levin, Peres & Wilmer, *Markov Chains and Mixing Times* (2017).
