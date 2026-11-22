"""
Mixing Time and the Cutoff Phenomenon
=====================================
An interactive explorer for how (and how suddenly) random processes become random.

Run with:   streamlit run app.py

Two exactly-computed examples (no Monte Carlo error in the distance curves):
  1. The lazy random walk on the d-dimensional hypercube.
  2. The Gilbert-Shannon-Reeds riffle shuffle of an n-card deck.
"""
from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Edit these two lines for your submission
# --------------------------------------------------------------------------- #
AUTHOR = "Ethan Tai"
COURSE = "Probability  ·  Washington and Lee University"

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, SKY, GREEN = "#0072B2", "#E69F00", "#56B4E9", "#009E73"
VERM, PURPLE, GREY, INK = "#D55E00", "#CC79A7", "#7A7A7A", "#1F2933"
SERIES = [BLUE, ORANGE, GREEN, VERM, PURPLE, SKY, GREY]

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

# --------------------------------------------------------------------------- #
# Mathematics: hypercube walk
# --------------------------------------------------------------------------- #
# Walk: at each step pick one of the d coordinates uniformly and re-set it with
# a fair coin flip.  Started at the all-zeros corner, the walk is symmetric under
# permuting coordinates, so its total-variation distance to uniform equals the
# distance between the *Hamming weight* (number of ones) and a Binomial(d, 1/2).
# That reduces a 2^d-state chain to a (d+1)-state chain we can iterate exactly.


def _binom_pmf(d: int) -> np.ndarray:
    logp = np.array(
        [math.lgamma(d + 1) - math.lgamma(k + 1) - math.lgamma(d - k + 1) for k in range(d + 1)]
    ) - d * math.log(2)
    return np.exp(logp)


def _weight_step(p: np.ndarray, down: np.ndarray, up: np.ndarray) -> np.ndarray:
    new = 0.5 * p                      # half the time the coin leaves the state alone
    new[:-1] += p[1:] * down[1:]       # a 1 gets turned into a 0   (w -> w-1)
    new[1:] += p[:-1] * up[:-1]        # a 0 gets turned into a 1   (w -> w+1)
    return new


def _weight_moves(d: int):
    w = np.arange(d + 1)
    return w / (2 * d), (d - w) / (2 * d)


@st.cache_data(show_spinner=False)
def hypercube_tv(d: int, t_max: int) -> np.ndarray:
    """Exact TV distance to uniform for t = 0..t_max."""
    down, up = _weight_moves(d)
    target = _binom_pmf(d)
    p = np.zeros(d + 1)
    p[0] = 1.0
    tv = np.empty(t_max + 1)
    tv[0] = 0.5 * np.abs(p - target).sum()
    for t in range(1, t_max + 1):
        p = _weight_step(p, down, up)
        tv[t] = 0.5 * np.abs(p - target).sum()
    return tv


@st.cache_data(show_spinner=False)
def hypercube_dist(d: int, t: int) -> np.ndarray:
    """Distribution of the number of ones after t steps."""
    down, up = _weight_moves(d)
    p = np.zeros(d + 1)
    p[0] = 1.0
    for _ in range(t):
        p = _weight_step(p, down, up)
    return p


def hypercube_cutoff(d: int) -> float:
    return 0.5 * d * math.log(d)


# --------------------------------------------------------------------------- #
# Mathematics: riffle shuffle
# --------------------------------------------------------------------------- #
# Bayer & Diaconis (1992): after k riffle shuffles, a permutation with r "rising
# sequences" has probability  C(2^k + n - r, n) / 2^(k n).  The number of
# permutations with r rising sequences is the Eulerian number A(n, r-1).
# Summing |P(pi) - 1/n!| over all permutations therefore collapses to n terms.


@lru_cache(maxsize=None)
def eulerian_row(n: int) -> tuple[int, ...]:
    """A(n, m) = number of permutations of n items with m descents, m = 0..n-1."""
    row = [1]
    for size in range(2, n + 1):
        new = [0] * size
        for m in range(size):
            a = (m + 1) * row[m] if m < len(row) else 0
            b = (size - m) * row[m - 1] if m >= 1 else 0
            new[m] = a + b
        row = new
    return tuple(row)


@lru_cache(maxsize=None)
def riffle_tv(n: int, k: int) -> float:
    """Exact TV distance to uniform after k riffle shuffles of n cards."""
    fact = math.factorial(n)
    denom = 2 ** (k * n)
    row = eulerian_row(n)
    total = 0
    for r in range(1, n + 1):
        c = math.comb(2**k + n - r, n)
        total += row[r - 1] * abs(c * fact - denom)
    return float(Fraction(total, 2 * denom * fact))


def riffle_cutoff(n: int) -> float:
    return 1.5 * math.log2(n)


def gsr_shuffle(deck: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One Gilbert-Shannon-Reeds riffle: binomial cut, then proportional interleave."""
    n = len(deck)
    cut = rng.binomial(n, 0.5)
    left, right = deck[:cut], deck[cut:]
    out = np.empty(n, dtype=deck.dtype)
    i = j = 0
    for pos in range(n):
        remaining_l, remaining_r = len(left) - i, len(right) - j
        if rng.random() < remaining_l / (remaining_l + remaining_r):
            out[pos] = left[i]
            i += 1
        else:
            out[pos] = right[j]
            j += 1
    return out


@st.cache_data(show_spinner=False)
def simulate_riffles(n: int, k_max: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    deck = np.arange(n)
    rows = [deck.copy()]
    for _ in range(k_max):
        deck = gsr_shuffle(deck, rng)
        rows.append(deck.copy())
    return np.array(rows)


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp, .stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp label, .stApp button, .stApp div[data-testid="stMetricValue"] {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}
.block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem; }
#MainMenu, footer { visibility: hidden; }

.hero { padding: 1.6rem 1.8rem; border-radius: 14px; margin-bottom: 1.4rem;
        background: linear-gradient(135deg, #F2F8FD 0%, #FFFFFF 70%);
        border: 1px solid #DCE7F1; border-left: 7px solid %BLUE%; }
.hero h1 { margin: 0 0 .35rem 0; font-size: 2.05rem; font-weight: 700; color: %INK%; letter-spacing: -0.02em; }
.hero .tag { font-size: 1.04rem; color: #4A5563; margin: 0 0 .7rem 0; max-width: 46rem; }
.hero .meta { font-size: .82rem; color: #7B8794; letter-spacing: .04em; text-transform: uppercase; }

h3 { font-weight: 650 !important; letter-spacing: -0.01em; color: %INK%; margin-top: .4rem; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #E4E9EF; }
.stTabs [data-baseweb="tab"] { height: 46px; padding: 0 18px; font-weight: 600; }

.callout { border-left: 5px solid %BLUE%; background: #F7F9FB; padding: .85rem 1.1rem;
           border-radius: 0 10px 10px 0; margin: .9rem 0 1.1rem 0; color: #2B3540; font-size: .96rem; }
.callout .ct { font-weight: 700; font-size: .78rem; letter-spacing: .07em; text-transform: uppercase;
               margin-bottom: .25rem; color: #52606D; }

div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E4E9EF; border-radius: 12px;
                              padding: .85rem 1rem; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
div[data-testid="stMetricLabel"] { color: #52606D; }
.small-note { font-size: .84rem; color: #7B8794; }
.footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #E4E9EF;
          font-size: .82rem; color: #7B8794; }
</style>
""".replace("%BLUE%", BLUE).replace("%INK%", INK)


def callout(title: str, body: str, color: str = BLUE) -> None:
    st.markdown(
        f'<div class="callout" style="border-left-color:{color}">'
        f'<div class="ct">{title}</div>{body}</div>',
        unsafe_allow_html=True,
    )


def show(fig: go.Figure) -> None:
    # `width="stretch"` is the current Streamlit spelling; fall back for older versions.
    try:
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CONFIG)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None, config=PLOT_CONFIG)


def style(fig: go.Figure, title: str, xtitle: str, ytitle: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color=INK)),
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=INK),
        margin=dict(l=8, r=8, t=64, b=8),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        hoverlabel=dict(font_family="Inter, sans-serif"),
    )
    fig.update_xaxes(title=xtitle, gridcolor="#EEF1F4", zeroline=False, showline=True, linecolor="#CBD2D9")
    fig.update_yaxes(title=ytitle, gridcolor="#EEF1F4", zeroline=False, showline=True, linecolor="#CBD2D9")
    return fig


# --------------------------------------------------------------------------- #
# Page sections
# --------------------------------------------------------------------------- #
def render_hero() -> None:
    st.markdown(
        f"""
<div class="hero">
  <h1>How Long Until Random Is Random?</h1>
  <p class="tag">An interactive look at <b>mixing times</b> and the <b>cutoff phenomenon</b>: why some
  processes stay obviously non-random for a long time and then become random almost all at once.</p>
  <div class="meta">{AUTHOR} &nbsp;·&nbsp; {COURSE}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_big_idea() -> None:
    st.markdown("### The big idea, in plain language")
    st.markdown(
        """
You shuffle a deck of cards. After one shuffle, it is still mostly in order. After a hundred, it is surely
scrambled. But **how many shuffles does it take for the deck to be "random enough"?** Mathematicians turned that
everyday question into a precise one, and the answer turned out to be surprising.
"""
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Four ideas you need")
        st.markdown(
            """
**1. A random process (Markov chain).** Something that moves step by step, where the next move depends only on
where you are *now*, not on how you got there. A shuffle is one: the new order depends only on the current order
and the luck of this shuffle.

**2. "Fully random."** Every possible arrangement is equally likely. For 52 cards that is
about 8 × 10⁶⁷ arrangements, all with the same chance.

**3. Distance from random (total variation).** Imagine a detective who is shown a deck and must guess:
*"Did this come from the shuffling process, or from a perfectly random deck?"* The total variation distance
measures how good the detective can possibly be.
- **1** means the detective is never wrong. The process is nowhere near random.
- **0** means the detective is no better than a coin flip. The process is perfectly random.

**4. Mixing time.** The number of steps until that distance falls below a small cutoff, such as ¼.
"""
        )
    with c2:
        st.markdown("#### The surprise: the cutoff")
        st.markdown(
            """
You might expect the distance to fade away gradually, like a hot drink slowly cooling.

For many processes it does not. The distance **stays close to 1 for a long time, then drops to
nearly 0 within a short window.** It behaves more like a **light switch** than a dimmer.

This is called the **cutoff phenomenon**. It means "almost random" and "obviously not random" are separated by
only a thin slice of time. In the plots below you will see that slice get thinner as the system gets bigger.
"""
        )
        callout(
            "Why it matters",
            "Cutoff means that a few extra shuffles, or one fewer, can be the difference between a deck "
            "a card counter can still exploit and one that is effectively random.",
            GREEN,
        )

    st.markdown("---")
    st.markdown("#### Our two examples, in everyday terms")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Example 1: A row of light switches (the hypercube walk)**")
        st.markdown(
            """
There are **d** switches, all off. Each step, pick one switch at random and set it by flipping a fair coin.

*Why a cliff?* A switch you have never picked is **still off**. If many switches are untouched, the row has
suspiciously few "on" switches, and the detective spots it. After *t* steps, about **d·e^(−t/d)** switches are
still untouched. Even a truly random row wobbles by about **√d** in how many switches are on, so the detective
gives up once the untouched count shrinks to that wobble size:

d·e^(−t/d) ≈ √d  ⟹  **t ≈ ½ · d · ln d**

Before that time the detective wins; after it, nearly every switch has been refreshed and they lose.
"""
        )
    with e2:
        st.markdown("**Example 2: Shuffling cards (the riffle shuffle)**")
        st.markdown(
            """
Cut the deck roughly in half, then let the two piles interleave, like when you riffle cards by thumb.

One riffle leaves a trace: you can still see **two ordered stretches** (cards from each pile stay in their original
relative order). After *k* riffles, the deck is a blend of up to **2ᵏ** ordered stretches. A random deck has
the fine-grained look of about *n*/2 of them, and matching it closely takes more than just reaching that count.

The famous result (Bayer and Diaconis, 1992) is that a deck of **n** cards needs about **1.5 · log₂ n** shuffles.
For 52 cards that is roughly **7 to 10 shuffles**, depending on how close to random you demand. Fewer is not
enough, and the distance drops quickly right around there.
"""
        )

    st.markdown(
        '<p class="small-note">Tip: open the next two tabs. Each has live sliders, an exact distance curve, '
        "and a side-by-side comparison showing the cliff getting sharper as the system grows.</p>",
        unsafe_allow_html=True,
    )


def render_hypercube() -> None:
    st.markdown("### Example 1: Random walk on the hypercube")
    st.markdown(
        "A row of **d light switches**, all starting *off*. Each step: pick one switch at random and "
        "re-set it with a fair coin flip. The distance to \"perfectly random\" is computed **exactly**, not simulated."
    )

    c1, c2 = st.columns(2)
    d = c1.slider("Number of switches (d)", min_value=4, max_value=400, value=64, step=1)
    frac = c2.slider(
        "Time shown, as a multiple of the cutoff time (½·d·ln d)",
        min_value=0.0, max_value=2.0, value=0.5, step=0.05,
    )

    t_star = hypercube_cutoff(d)
    t_max = int(math.ceil(2 * t_star))
    tv = hypercube_tv(d, t_max)
    t = int(round(frac * t_star))
    t = min(t, t_max)
    t_mix = int(np.argmax(tv < 0.25)) if (tv < 0.25).any() else t_max

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted cutoff time  ½·d·ln d", f"{t_star:,.0f} steps")
    m2.metric(f"Distance from random at step {t:,}", f"{tv[t]:.3f}")
    m3.metric("Steps until distance < 0.25", f"{t_mix:,}", delta=f"{t_mix / t_star:.2f} × cutoff time", delta_color="off")

    # ---- Plot A: TV curve ------------------------------------------------- #
    ts = np.arange(t_max + 1)
    fig = go.Figure()
    fig.add_vrect(x0=max(0, t_star - d), x1=t_star + d, fillcolor=ORANGE, opacity=0.13, line_width=0,
                  annotation_text="cutoff window (± d)", annotation_position="top left",
                  annotation_font=dict(size=11, color="#8A5A00"))
    fig.add_vline(x=t_star, line=dict(color=ORANGE, width=2, dash="dash"))
    fig.add_trace(go.Scatter(x=ts, y=tv, mode="lines", name="Distance from random",
                             line=dict(color=BLUE, width=3), hovertemplate="step %{x}<br>distance %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[t], y=[tv[t]], mode="markers", name="Time shown",
                             marker=dict(color=VERM, size=12, line=dict(color="white", width=2))))
    style(fig, f"With {d} switches, the distance stays near 1, then collapses around step {t_star:,.0f}",
          "Steps taken", "Total variation distance from random")
    fig.update_yaxes(range=[0, 1.04])
    show(fig)

    # ---- Plot B: weight distribution ---------------------------------------- #
    p = hypercube_dist(d, t)
    target = _binom_pmf(d)
    w = np.arange(d + 1)
    mean_on = float((w * p).sum())
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=w, y=p, name=f"After {t:,} steps", marker_color=BLUE, opacity=0.85))
    fig2.add_trace(go.Scatter(x=w, y=target, mode="lines", name="Perfectly random row",
                              line=dict(color=ORANGE, width=3)))
    style(fig2,
          f"What the detective sees: about {mean_on:.1f} switches on (a random row averages {d / 2:.0f})",
          "Number of switches that are on", "Probability", height=380)
    show(fig2)
    callout(
        "What to notice",
        "At time 0 every switch is off, so the blue spike sits at 0. As time passes the bump slides toward the "
        "orange curve. The distance <i>can only be small once the blue bars sit on top of the orange curve</i>. "
        "Slide the time control through 1.0 (the cutoff time) and watch that overlap snap into place.",
    )

    # ---- Plot C: cutoff sharpens -------------------------------------------- #
    st.markdown("#### The cutoff gets sharper as the cube grows")
    dims = st.multiselect("Compare sizes (number of switches)", [8, 16, 32, 64, 128, 256, 512],
                          default=[8, 32, 128, 512], key="hc_dims")
    fig3 = go.Figure()
    for i, dd in enumerate(sorted(dims)):
        ts_star = hypercube_cutoff(dd)
        tv_dd = hypercube_tv(dd, int(math.ceil(2 * ts_star)))
        x = np.arange(len(tv_dd)) / ts_star
        fig3.add_trace(go.Scatter(x=x, y=tv_dd, mode="lines", name=f"d = {dd}",
                                  line=dict(color=SERIES[i % len(SERIES)], width=2.6)))
    fig3.add_vline(x=1, line=dict(color=GREY, width=1.5, dash="dot"))
    style(fig3, "Measured in units of ½·d·ln d, bigger cubes drop more and more like a step function",
          "Time ÷ (½ · d · ln d)", "Total variation distance from random")
    fig3.update_yaxes(range=[0, 1.04])
    show(fig3)
    callout(
        "What to notice",
        "All the curves cross near 1 on the horizontal axis (the dotted line), and the larger the cube, the more "
        "vertical the drop. Honest caveat: the window is about <i>d</i> steps wide while the cutoff time is "
        "½·<i>d</i>·ln <i>d</i>, so the sharpening is only logarithmic in <i>d</i>, which is why it looks gradual.",
        ORANGE,
    )


def render_shuffle() -> None:
    st.markdown("### Example 2: Riffle-shuffling a deck")
    st.markdown(
        "Cut the deck about in half and interleave the two piles (the Gilbert–Shannon–Reeds model of a real "
        "riffle shuffle). The distance to a perfectly random deck is computed **exactly** with the Bayer–Diaconis formula."
    )

    n = st.slider("Number of cards in the deck (n)", min_value=4, max_value=208, value=52, step=1)
    t_star = riffle_cutoff(n)
    k_max = max(12, int(math.ceil(2 * t_star)) + 1)
    ks = np.arange(k_max + 1)
    tv = np.array([riffle_tv(n, int(k)) for k in ks])

    def first_below(thr: float) -> int:
        idx = np.where(tv < thr)[0]
        return int(idx[0]) if len(idx) else k_max

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted cutoff  1.5·log₂ n", f"{t_star:.1f} shuffles")
    m2.metric("Distance < 0.5 after", f"{first_below(0.5)} shuffles")
    m3.metric("Distance < 0.25 after", f"{first_below(0.25)} shuffles")
    m4.metric("Distance < 0.1 after", f"{first_below(0.1)} shuffles")

    fig = go.Figure()
    fig.add_vline(x=t_star, line=dict(color=ORANGE, width=2, dash="dash"),
                  annotation_text="1.5 · log₂ n", annotation_position="top right",
                  annotation_font=dict(size=11, color="#8A5A00"))
    fig.add_trace(go.Scatter(x=ks, y=tv, mode="lines+markers", name="Distance from random",
                             line=dict(color=BLUE, width=3), marker=dict(size=8, color=BLUE),
                             hovertemplate="%{x} shuffles<br>distance %{y:.4f}<extra></extra>"))
    style(fig, f"A {n}-card deck is far from random for several shuffles, then the distance falls quickly",
          "Number of riffle shuffles", "Total variation distance from random")
    fig.update_yaxes(range=[0, 1.04])
    fig.update_xaxes(dtick=1 if k_max <= 25 else None)
    show(fig)

    # ---- Heat-strip of simulated shuffles ----------------------------------- #
    st.markdown("#### See it: a deck being shuffled")
    st.markdown(
        "Each row is the deck after that many shuffles. Colour shows each card's **original position** "
        "(a smooth gradient means \"still in order\"). Watch the pattern dissolve into static."
    )
    if "seed" not in st.session_state:
        st.session_state.seed = 7
    b1, b2 = st.columns([1, 5])
    if b1.button("Reshuffle", help="Draw a new set of random shuffles"):
        st.session_state.seed += 1
    rows_shown = min(k_max, 10)
    sim = simulate_riffles(n, rows_shown, st.session_state.seed)
    heat = go.Figure(go.Heatmap(
        z=sim, x=np.arange(1, n + 1), y=np.arange(rows_shown + 1), colorscale="Viridis",
        colorbar=dict(title="Original<br>position", thickness=14),
        hovertemplate="after %{y} shuffles<br>deck position %{x}<br>original card #%{z}<extra></extra>",
    ))
    style(heat, "Structure disappears as shuffles accumulate", "Position in the deck", "Shuffles", height=430)
    heat.update_yaxes(autorange="reversed", dtick=1)
    show(heat)
    callout(
        "What to notice",
        "After 1 or 2 shuffles you can still read off long smooth stripes (ordered runs). Around "
        f"{t_star:.0f} to {t_star + 2:.0f} shuffles for this deck size, the stripes are too fine for the eye "
        "to find, which matches where the distance curve above falls off its cliff.",
    )

    # ---- Comparison across deck sizes ---------------------------------------- #
    st.markdown("#### The cutoff gets sharper as the deck grows")
    sizes = st.multiselect("Compare deck sizes", [8, 16, 32, 52, 104, 208],
                           default=[8, 16, 52, 208], key="deck_sizes")
    fig3 = go.Figure()
    for i, nn in enumerate(sorted(sizes)):
        ts = riffle_cutoff(nn)
        kk = np.arange(int(math.ceil(2 * ts)) + 2)
        yy = [riffle_tv(nn, int(k)) for k in kk]
        fig3.add_trace(go.Scatter(x=kk / ts, y=yy, mode="lines+markers", name=f"n = {nn}",
                                  line=dict(color=SERIES[i % len(SERIES)], width=2.4), marker=dict(size=6)))
    fig3.add_vline(x=1, line=dict(color=GREY, width=1.5, dash="dot"))
    style(fig3, "In units of 1.5·log₂ n, every deck size drops at about the same place",
          "Shuffles ÷ (1.5 · log₂ n)", "Total variation distance from random")
    fig3.update_yaxes(range=[0, 1.04])
    show(fig3)


def render_notes() -> None:
    st.markdown("### Math notes (for the curious)")

    st.markdown("**Total variation distance.** For distributions μ and π on a finite set Ω:")
    st.latex(r"\|\mu-\pi\|_{TV}=\max_{A\subseteq\Omega}\,|\mu(A)-\pi(A)|=\tfrac12\sum_{x\in\Omega}|\mu(x)-\pi(x)|")
    st.markdown("**Mixing time.** For a Markov chain with transition law $P^t(x_0,\\cdot)$ and stationary law $\\pi$:")
    st.latex(r"d(t)=\|P^t(x_0,\cdot)-\pi\|_{TV},\qquad t_{\text{mix}}(\varepsilon)=\min\{t: d(t)\le\varepsilon\}")
    st.markdown("**Cutoff.** A family of chains (indexed by size $n$) exhibits cutoff if, for every fixed $\\varepsilon\\in(0,1)$,")
    st.latex(r"\frac{t^{(n)}_{\text{mix}}(\varepsilon)}{t^{(n)}_{\text{mix}}(1-\varepsilon)}\longrightarrow 1\quad (n\to\infty).")
    st.markdown("In words: the time to get from distance $1-\\varepsilon$ to $\\varepsilon$ is negligible compared with the time to get started.")

    st.markdown("---")
    st.markdown("#### Hypercube walk")
    st.markdown(
        "Lazy walk on $\\{0,1\\}^d$: choose a coordinate uniformly and replace it by a fair bit. "
        "Starting at the all-zeros corner, symmetry lets us track only the Hamming weight $W_t$. "
        "That is a birth-death chain on $\\{0,\\dots,d\\}$ with"
    )
    st.latex(r"P(w\to w-1)=\frac{w}{2d},\qquad P(w\to w+1)=\frac{d-w}{2d},\qquad P(w\to w)=\tfrac12,")
    st.markdown(
        "and the TV distance of the full walk equals the TV distance of $W_t$ to $\\mathrm{Bin}(d,\\tfrac12)$. "
        "This is exactly what the app iterates. The known result (Levin–Peres–Wilmer, *Markov Chains and Mixing Times*) is"
    )
    st.latex(r"t_{\text{mix}}(\varepsilon)=\tfrac12\,d\ln d+O_\varepsilon(d),")
    st.markdown("so there is cutoff at $\\tfrac12 d\\ln d$ with window of order $d$.")

    st.markdown("#### Riffle shuffle")
    st.markdown(
        "Under the Gilbert–Shannon–Reeds model, $k$ riffle shuffles act like a single $a$-shuffle with $a=2^k$. "
        "If a permutation $\\pi$ has $r$ rising sequences, then"
    )
    st.latex(r"P_k(\pi)=\frac{1}{2^{kn}}\binom{2^k+n-r}{n}.")
    st.markdown(
        "The number of permutations with $r$ rising sequences is the Eulerian number $A(n,r-1)$, so"
    )
    st.latex(r"\|P_k-U\|_{TV}=\frac12\sum_{r=1}^{n}A(n,r-1)\left|\frac{1}{2^{kn}}\binom{2^k+n-r}{n}-\frac1{n!}\right|.")
    st.markdown("Bayer and Diaconis showed that for $k=\\tfrac32\\log_2 n+c$, as $n\\to\\infty$,")
    st.latex(r"\|P_k-U\|_{TV}\to 1-2\Phi\!\left(\frac{-2^{-c}}{4\sqrt3}\right),")
    st.markdown("where $\\Phi$ is the standard normal CDF. So there is cutoff at $\\tfrac32\\log_2 n$ with a window of constant width.")

    st.markdown("#### References")
    st.markdown(
        """
- D. Aldous and P. Diaconis, *Shuffling cards and stopping times*, Amer. Math. Monthly 93 (1986).
- D. Bayer and P. Diaconis, *Trailing the dovetail shuffle to its lair*, Ann. Appl. Probab. 2 (1992).
- P. Diaconis, *The cutoff phenomenon in finite Markov chains*, PNAS 93 (1996).
- D. Levin, Y. Peres, E. Wilmer, *Markov Chains and Mixing Times*, 2nd ed., AMS (2017).
"""
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="How Long Until Random Is Random?", page_icon="🎲", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    render_hero()

    tab1, tab2, tab3, tab4 = st.tabs(["The big idea", "Hypercube walk", "Card shuffling", "Math notes"])
    with tab1:
        render_big_idea()
    with tab2:
        render_hypercube()
    with tab3:
        render_shuffle()
    with tab4:
        render_notes()

    st.markdown(
        f'<div class="footer">{AUTHOR} · {COURSE} · Distances are computed exactly; '
        "only the card-heatmap uses random simulation (set by the Reshuffle button).</div>",
        unsafe_allow_html=True,
    )


main()
