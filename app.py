# -*- coding: utf-8 -*-
"""
Laboratorio Interactivo de Cálculo II
Aplicación Streamlit de estudio y verificación (9 módulos).
"""
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sympy as sp
import streamlit as st
import plotly.graph_objects as go
from scipy import ndimage, optimize
from scipy import integrate as sci_int
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication,
    function_exponentiation, convert_xor,
)

APP_TITLE = "Laboratorio Interactivo de Cálculo II"

# ----------------------------------------------------------------------------
# 1. UTILIDADES GENERALES: símbolos, parser, formato LaTeX, cómputo seguro
# ----------------------------------------------------------------------------
x, y, t = sp.symbols("x y t", real=True)

_LOCALS = {
    "x": x, "y": y, "t": t, "pi": sp.pi, "e": sp.E, "E": sp.E, "oo": sp.oo,
    "sen": sp.sin, "sin": sp.sin, "cos": sp.cos, "tg": sp.tan, "tan": sp.tan,
    "cotg": sp.cot, "cot": sp.cot, "sec": sp.sec, "cosec": sp.csc, "csc": sp.csc,
    "arcsen": sp.asin, "arcsin": sp.asin, "asin": sp.asin,
    "arccos": sp.acos, "acos": sp.acos,
    "arctg": sp.atan, "arctan": sp.atan, "atan": sp.atan,
    "arccotg": sp.acot, "arccot": sp.acot,
    "senh": sp.sinh, "sinh": sp.sinh, "cosh": sp.cosh,
    "tgh": sp.tanh, "tanh": sp.tanh, "cotgh": sp.coth, "coth": sp.coth,
    "ln": sp.log, "log": sp.log, "exp": sp.exp,
    "sqrt": sp.sqrt, "raiz": sp.sqrt, "abs": sp.Abs,
    "cbrt": lambda v: v ** sp.Rational(1, 3),
    "logb": lambda b, v: sp.log(v) / sp.log(b),
}
_TRANSF = standard_transformations + (
    implicit_multiplication, function_exponentiation, convert_xor)
_PERMITIDOS = re.compile(r"^[0-9A-Za-z_+\-*/^()., \t]*$")


def parse_math(texto, vars_ok=(x,)):
    """Convierte texto del usuario en una expresión SymPy (con validación)."""
    s = (texto or "").strip()
    if not s:
        raise ValueError("Hay un campo vacío.")
    s = (s.replace("−", "-").replace("·", "*").replace("×", "*")
          .replace("π", "pi").replace("∞", "oo"))
    s = re.sub(r"√\s*\(", "sqrt(", s)
    s = re.sub(r"√\s*([A-Za-z0-9.]+)", r"sqrt(\1)", s)
    if "__" in s or not _PERMITIDOS.match(s):
        raise ValueError(f"Caracteres no permitidos en «{texto}».")
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    s = re.sub(r"\blog_(\d+(?:\.\d+)?|[a-z])\(", r"logb(\1,", s)
    s = re.sub(r"\blog(\d+)\(", r"logb(\1,", s)
    try:
        expr = parse_expr(s, local_dict=dict(_LOCALS), transformations=_TRANSF)
    except Exception as exc:  # noqa
        raise ValueError(
            f"No pude interpretar «{texto}». Usá * entre factores y paréntesis "
            f"en las funciones, por ejemplo sen(2*x). ({type(exc).__name__})")
    expr = sp.sympify(expr)
    extra = [str(v) for v in expr.free_symbols if v not in vars_ok]
    if extra:
        raise ValueError(
            f"Variable no permitida en «{texto}»: {', '.join(extra)}. "
            f"Variables válidas: {', '.join(str(v) for v in vars_ok)}.")
    return expr


_LATEX_ES = [
    ("\\sinh", "\\operatorname{senh}"), ("\\tanh", "\\operatorname{tgh}"),
    ("\\coth", "\\operatorname{cotgh}"), ("\\arcsin", "\\operatorname{arcsen}"),
    ("\\arctan", "\\operatorname{arctg}"), ("\\sin", "\\operatorname{sen}"),
    ("\\tan", "\\operatorname{tg}"), ("\\cot", "\\operatorname{cotg}"),
    ("\\csc", "\\operatorname{cosec}"),
]


def L(expr):
    """LaTeX con nomenclatura de la cátedra (sen, tg, arctg, ...)."""
    try:
        s = sp.latex(sp.sympify(expr), inv_trig_style="full", ln_notation=True)
    except Exception:
        s = str(expr)
    for a, b in _LATEX_ES:
        s = s.replace(a, b)
    return s


def N6(v):
    try:
        return f"{float(v):.6g}"
    except Exception:
        return str(v)


_POOL = ThreadPoolExecutor(max_workers=8)


def con_limite(fn, *args, timeout=10, default=None):
    """Ejecuta fn con límite de tiempo; si falla o se demora devuelve default."""
    fut = _POOL.submit(fn, *args)
    try:
        return fut.result(timeout=timeout)
    except Exception:
        return default


def _es_raiz_impar(e):
    return (e.is_Pow and e.exp.is_Rational and e.exp.q > 1 and e.exp.q % 2 == 1
            and e.base.free_symbols)


def para_numerico(expr):
    """Raíces de índice impar con base negativa reales (p. ej. ∛(-8) = -2)."""
    expr = sp.sympify(expr)
    if not any(_es_raiz_impar(p) for p in expr.atoms(sp.Pow)):
        return expr
    return expr.replace(
        _es_raiz_impar,
        lambda e: sp.sign(e.base) ** e.exp.p * sp.Abs(e.base) ** e.exp)


def _post(r, shape):
    r = np.asarray(r)
    if r.shape != shape:
        r = np.broadcast_to(r, shape)
    if np.iscomplexobj(r):
        r = np.where(np.abs(r.imag) < 1e-9 * (1 + np.abs(r.real)), r.real, np.nan)
    r = r.astype(float).copy()
    r[~np.isfinite(r)] = np.nan
    return r


def num_fn(expr, var=x):
    """Función numérica vectorizada (NaN donde no está definida)."""
    f = sp.lambdify(var, para_numerico(expr), modules=["numpy"])

    def g(v):
        v = np.asarray(v, dtype=float)
        with np.errstate(all="ignore"):
            try:
                r = f(v)
            except Exception:
                r = np.full(v.shape, np.nan)
        return _post(r, v.shape)
    return g


def num_fn2(expr):
    f = sp.lambdify((x, y), para_numerico(expr), modules=["numpy"])

    def g(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        shp = np.broadcast(a, b).shape
        with np.errstate(all="ignore"):
            try:
                r = f(a, b)
            except Exception:
                r = np.full(shp, np.nan)
        return _post(r, shp)
    return g


def exactificar(v, tol=1e-11, simple=False):
    """Reconoce un valor numérico como exacto: racional, múltiplo de π o e,
    e^k, raíz cuadrada de un racional o logaritmo de un racional.
    Con simple=True sólo acepta formas muy sencillas (para valores aproximados)."""
    from fractions import Fraction
    try:
        v = float(v)
    except Exception:
        return v
    tv = tol * max(1.0, abs(v))
    if abs(v - round(v)) < tv:
        return sp.Integer(int(round(v)))
    fr = Fraction(v).limit_denominator(24 if simple else 2000)
    if abs(float(fr) - v) < tv:
        return sp.Rational(fr.numerator, fr.denominator)
    for const, val in ((sp.pi, np.pi), (sp.E, np.e)):
        r = Fraction(v / val).limit_denominator(24 if simple else 240)
        if abs(float(r) * val - v) < tv:
            return sp.Rational(r.numerator, r.denominator) * const
    if simple:
        return sp.Float(v, 15)
    for k in range(-8, 9):
        if k and abs(np.exp(k) - v) < tv:
            return sp.exp(k)
    if v > 0:
        r = Fraction(v * v).limit_denominator(1000)
        if abs(float(r) - v * v) < 1e-10 * max(1.0, v * v):
            return sp.sqrt(sp.Rational(r.numerator, r.denominator))
        ev = np.exp(v)
        r = Fraction(ev).limit_denominator(200)
        if abs(float(r) - ev) < 1e-10 * max(1.0, ev) and r.numerator < 5000:
            return sp.log(sp.Rational(r.numerator, r.denominator))
    return sp.Float(v, 15)


def snap_frontera(v):
    """Valor de frontera de dominio (hallado por bisección) como número exacto."""
    for tol, simple in ((1e-11, False), (1e-8, True), (1e-6, True)):
        e = exactificar(v, tol, simple)
        if not isinstance(e, sp.Float):
            return e
    return sp.Float(float(v), 15)


def limpiar_exacto(e):
    """Presenta resultados exactos de forma legible (logaritmos de enteros)."""
    try:
        e = sp.expand_log(sp.sympify(e), force=True)
        e = e.replace(
            lambda q: q.func == sp.log and q.args[0].is_Integer and q.args[0] > 3,
            lambda q: sum(k * sp.log(pr) for pr, k in
                          sp.factorint(int(q.args[0])).items()))
        return sp.expand_log(sp.nsimplify(e) if False else e, force=True)
    except Exception:
        return e


# ---- integración definida robusta -----------------------------------------
def _fix_raices_impares(expr, var, xm):
    def rep(e):
        try:
            v = float(e.base.subs(var, xm))
        except Exception:
            return e
        if not np.isfinite(v) or v >= 0:
            return e
        return sp.Integer(-1) ** e.exp.p * (-e.base) ** e.exp
    return expr.replace(_es_raiz_impar, rep)


def _fix_abs(expr, var, a, b):
    if not expr.has(sp.Abs):
        return expr
    pts = np.linspace(a, b, 41)[1:-1]
    for ab in sorted(expr.atoms(sp.Abs), key=lambda q: -sp.count_ops(q)):
        arg = ab.args[0]
        if var not in arg.free_symbols:
            continue
        v = num_fn(arg, var)(pts)
        v = v[np.isfinite(v)]
        if v.size and np.all(v >= -1e-12):
            expr = expr.subs(ab, arg)
        elif v.size and np.all(v <= 1e-12):
            expr = expr.subs(ab, -arg)
    return expr


def _lim_num(v):
    v = sp.sympify(v)
    if v == sp.oo:
        return np.inf
    if v == -sp.oo:
        return -np.inf
    return float(sp.N(v))


def integrar_definida(expr, var, a, b):
    """Integral definida: valor exacto (si existe) y numérico."""
    expr, a, b = sp.sympify(expr), sp.sympify(a), sp.sympify(b)
    out = {"exacto": None, "num": None, "metodo": ""}
    try:
        am, bm = _lim_num(a), _lim_num(b)
    except Exception:
        am = bm = None
    ex2 = expr
    if am is not None and np.isfinite(am) and np.isfinite(bm):
        xm = (am + bm) / 2
        ex2 = _fix_abs(_fix_raices_impares(expr, var, xm), var, am, bm)
    res = con_limite(lambda: sp.integrate(ex2, (var, a, b)), timeout=14)
    if (res is not None and not res.has(sp.Integral) and not res.has(sp.I)
            and not res.has(sp.zoo) and not res.has(sp.nan)):
        simp = con_limite(lambda: sp.simplify(res), timeout=6, default=res)
        out["exacto"] = limpiar_exacto(simp)
        simp = out["exacto"]
        try:
            out["num"] = float(sp.N(simp))
        except Exception:
            pass
        out["metodo"] = "simbólico"
    if out["num"] is None and am is not None:
        fn = num_fn(expr, var)
        try:
            val, _err = sci_int.quad(
                lambda s: float(fn(np.array([s]))[0]), am, bm, limit=300)
            out["num"] = float(val)
            out["metodo"] = "numérico"
        except Exception:
            pass
    return out


def fmt_valor(res):
    if res["exacto"] is not None:
        return L(res["exacto"]), res["num"]
    return None, res["num"]


def fig_base(titulo="", alto=430):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=14)),
        height=alto, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.15),
        xaxis=dict(zeroline=True, zerolinecolor="#888", gridcolor="#e6e6e6",
                   title="x"),
        yaxis=dict(zeroline=True, zerolinecolor="#888", gridcolor="#e6e6e6",
                   title="y"),
        plot_bgcolor="white", hovermode="closest",
    )
    return fig


PLOT_CONFIG = {"scrollZoom": True, "displaylogo": False, "responsive": True}


def mostrar_fig(fig):
    try:
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
    except TypeError:  # versiones nuevas de Streamlit
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)


# ----------------------------------------------------------------------------
# 2. MOTOR DE CURVAS: intersecciones, regiones acotadas y barridos dx / dy
# ----------------------------------------------------------------------------
def _ramas(F, var):
    """Despeja la variable `var` de F=0 y devuelve las ramas reales."""
    sols = con_limite(lambda: sp.solve(F, var), timeout=7, default=[]) or []
    otra = y if var == x else x
    res = []
    for s in sols:
        if isinstance(s, dict):
            continue
        s = sp.sympify(s)
        if s.has(sp.I) or s.has(sp.Piecewise) or s.has(sp.zoo):
            continue
        if not s.free_symbols <= {otra}:
            continue
        if any(sp.simplify(s - r) == 0 for r in res):
            continue
        res.append(s)
    return res


class Curva:
    """Curva dada como  y = f(x),  x = g(y)  o relación implícita F(x,y)=0."""

    def __init__(self, texto, idx):
        self.idx = idx
        self.texto = texto.strip()
        if self.texto.count("=") != 1:
            raise ValueError(
                f"La curva «{texto}» debe escribirse como  y = ...  o  x = ...")
        izq, der = self.texto.split("=")
        Li = parse_math(izq, (x, y))
        Ri = parse_math(der, (x, y))
        self.F = sp.expand(Li - Ri)
        self.kind, self.expr = "impl", None
        if Li == y and y not in Ri.free_symbols:
            self.kind, self.expr = "y", Ri
        elif Ri == y and y not in Li.free_symbols:
            self.kind, self.expr = "y", Li
        elif Li == x and x not in Ri.free_symbols:
            self.kind, self.expr = "x", Ri
        elif Ri == x and x not in Li.free_symbols:
            self.kind, self.expr = "x", Li
        self._res = num_fn2(self.F)
        self._yb = None
        self._xb = None

    # -- ramas ---------------------------------------------------------
    def y_branches(self):
        if self._yb is None:
            if self.kind == "y":
                self._yb = [self.expr]
            elif self.kind == "x" and not self.expr.free_symbols:
                self._yb = []
            else:
                self._yb = _ramas(self.F, y)
        return self._yb

    def x_branches(self):
        if self._xb is None:
            if self.kind == "x":
                self._xb = [self.expr]
            elif self.kind == "y" and not self.expr.free_symbols:
                self._xb = []
            else:
                self._xb = _ramas(self.F, x)
        return self._xb

    @property
    def is_vline(self):
        return self.kind == "x" and not self.expr.free_symbols

    @property
    def is_hline(self):
        return self.kind == "y" and not self.expr.free_symbols

    @property
    def modo(self):
        if self.y_branches():
            return "y"
        if self.x_branches():
            return "x"
        return None

    # -- evaluación validada (descarta ramas espurias de la inversión) ---
    def ev_y(self, k, xs):
        xs = np.asarray(xs, dtype=float)
        v = num_fn(self.y_branches()[k], x)(xs)
        if self.kind != "y":
            r = self._res(xs, v)
            with np.errstate(all="ignore"):
                ok = np.abs(r) <= 1e-10 * (1 + xs ** 2 + v ** 2)
            v = np.where(ok, v, np.nan)
        return v

    def ev_x(self, k, ys):
        ys = np.asarray(ys, dtype=float)
        v = num_fn(self.x_branches()[k], y)(ys)
        if self.kind != "x":
            r = self._res(v, ys)
            with np.errstate(all="ignore"):
                ok = np.abs(r) <= 1e-10 * (1 + ys ** 2 + v ** 2)
            v = np.where(ok, v, np.nan)
        return v


def raices_1d(fd, lo, hi, n=6001):
    """Raíces numéricas de fd en [lo,hi] (cruces de signo y tangencias)."""
    g = np.linspace(lo, hi, n)
    v = fd(g)
    roots = []
    with np.errstate(all="ignore"):
        s = np.sign(v)
        idx = np.where(s[:-1] * s[1:] < 0)[0]
    for i in idx:
        try:
            r = optimize.brentq(lambda z: float(fd(np.array([z]))[0]),
                                g[i], g[i + 1], xtol=1e-14)
        except Exception:
            continue
        fr = fd(np.array([r]))[0]
        if np.isfinite(fr) and abs(fr) < 1e-6 * (1 + abs(r)):
            roots.append(r)
    roots += [float(g[i]) for i in np.where(v == 0)[0]]
    av = np.abs(v)
    with np.errstate(all="ignore"):
        cand = np.where((av[1:-1] < av[:-2]) & (av[1:-1] <= av[2:])
                        & np.isfinite(av[1:-1]))[0] + 1
    fin = av[np.isfinite(av)]
    escala = 1 + (np.median(fin) if fin.size else 0)
    for i in cand[:400]:
        if av[i] > 1e-2 * escala:
            continue
        try:
            r = optimize.minimize_scalar(
                lambda z: float(np.abs(fd(np.array([z]))[0])),
                bounds=(g[i - 1], g[i + 1]), method="bounded",
                options={"xatol": 1e-13})
            if r.fun < 1e-9:
                roots.append(float(r.x))
        except Exception:
            continue
    roots.sort()
    out = []
    for r in roots:
        if not out or abs(r - out[-1]) > 1e-7:
            out.append(r)
    return out


def raices_mixtas(fd, diff_expr, var, lo, hi):
    """Raíces: numéricas (robustas) + exactas (SymPy) si es posible."""
    nums = raices_1d(fd, lo, hi)
    exact = []
    if diff_expr is not None:
        sols = con_limite(lambda: sp.solve(diff_expr, var), timeout=5,
                          default=[]) or []
        for s in sols:
            if isinstance(s, dict):
                continue
            try:
                c = complex(sp.N(s))
            except Exception:
                continue
            if abs(c.imag) > 1e-9 or not (lo <= c.real <= hi):
                continue
            fv = fd(np.array([c.real]))[0]
            if np.isfinite(fv) and abs(fv) < 1e-7 * (1 + abs(c.real)):
                exact.append((sp.sympify(s), c.real))
    out = []
    for r in nums:
        m = [s for s, v in exact if abs(v - r) < 1e-7]
        out.append(m[0] if m else exactificar(r))
    for s, v in exact:
        if not any(abs(float(sp.N(q)) - v) < 1e-7 for q in out):
            out.append(s)
    return out


def _ffloat(v):
    return float(sp.N(v))


class Motor:
    """Analiza un conjunto de curvas: puntos de corte, región(es) acotada(s) y
    barridos con rectángulos diferenciales paralelos al eje y (dx) o x (dy)."""

    NR = 900  # resolución del raster de regiones

    def __init__(self, textos, ventana=30.0):
        self.curvas = [Curva(tx, i) for i, tx in enumerate(
            [q for q in textos if q.strip()])]
        if not self.curvas:
            raise ValueError("Ingresá al menos una curva.")
        for c in self.curvas:
            if c.modo is None:
                raise ValueError(
                    f"No pude despejar una variable de «{c.texto}». "
                    f"Escribila como y = f(x) o x = g(y).")
        self.win = float(ventana)
        vx = sorted(_ffloat(c.expr) for c in self.curvas if c.is_vline)
        hy = sorted(_ffloat(c.expr) for c in self.curvas if c.is_hline)
        self.banda_x, self.banda_y = len(vx) >= 2, len(hy) >= 2
        self.rx = (vx[0], vx[-1]) if self.banda_x else (-self.win, self.win)
        self.ry = (hy[0], hy[-1]) if self.banda_y else (-self.win, self.win)
        self._fr = {}
        self.puntos = self._intersecciones()
        self.caja = self._caja()
        self._enc = None

    # -- intersecciones --------------------------------------------------
    def _inter_par(self, A, B):
        W = self.win
        rx, ry = self.rx, self.ry
        pts = []
        mA, mB = A.modo, B.modo
        if mA == "y" and mB == "y":
            for i, ha in enumerate(A.y_branches()):
                for j, hb in enumerate(B.y_branches()):
                    if sp.simplify(ha - hb) == 0:
                        continue
                    fd = lambda g, i=i, j=j: A.ev_y(i, g) - B.ev_y(j, g)
                    for r in raices_mixtas(fd, ha - hb, x, rx[0], rx[1]):
                        pts.append((r, self._val(ha, x, r)))
        elif mA == "x" and mB == "x":
            for i, ga in enumerate(A.x_branches()):
                for j, gb in enumerate(B.x_branches()):
                    if sp.simplify(ga - gb) == 0:
                        continue
                    fd = lambda g, i=i, j=j: A.ev_x(i, g) - B.ev_x(j, g)
                    for r in raices_mixtas(fd, ga - gb, y, ry[0], ry[1]):
                        pts.append((self._val(ga, y, r), r))
        else:
            if mA == "x":
                A, B = B, A  # A es modo y, B es modo x
            if B.is_vline:
                c = B.expr
                for k in range(len(A.y_branches())):
                    v = A.ev_y(k, np.array([_ffloat(c)]))[0]
                    if np.isfinite(v):
                        pts.append((c, self._val(A.y_branches()[k], x, c)))
            elif A.is_hline:
                d = A.expr
                for k in range(len(B.x_branches())):
                    v = B.ev_x(k, np.array([_ffloat(d)]))[0]
                    if np.isfinite(v):
                        pts.append((self._val(B.x_branches()[k], y, d), d))
            else:
                for i in range(len(A.y_branches())):
                    for j in range(len(B.x_branches())):
                        fd = lambda g, i=i, j=j: A.ev_y(i, B.ev_x(j, g)) - g
                        for r in raices_1d(fd, ry[0], ry[1]):
                            xv = float(B.ev_x(j, np.array([r]))[0])
                            pts.append((exactificar(xv), exactificar(r)))
        return pts

    @staticmethod
    def _val(expr, var, r):
        """Evalúa expr en var=r de forma exacta si r es exacto."""
        try:
            v = sp.simplify(expr.subs(var, r)) if not isinstance(r, sp.Float) \
                else expr.subs(var, r)
            if v.has(sp.I) or v.has(sp.zoo) or v.has(sp.nan):
                raise ValueError
            if abs(complex(sp.N(v)).imag) > 1e-9:
                raise ValueError
            return v
        except Exception:
            return exactificar(float(num_fn(expr, var)(np.array([_ffloat(r)]))[0]))

    def _intersecciones(self):
        pts = []
        cs = self.curvas
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                try:
                    pts += self._inter_par(cs[i], cs[j])
                except Exception:
                    continue
        # limpiar: reales, finitos y sin duplicados
        out = []
        for px, py in pts:
            try:
                fx, fy = _ffloat(px), _ffloat(py)
            except Exception:
                continue
            if not (np.isfinite(fx) and np.isfinite(fy)):
                continue
            if any(abs(fx - q[2]) < 1e-7 and abs(fy - q[3]) < 1e-7 for q in out):
                continue
            out.append((sp.sympify(px), sp.sympify(py), fx, fy))
        out.sort(key=lambda q: (q[2], q[3]))
        return out

    # -- caja de trabajo ----------------------------------------------------
    def _caja(self):
        xs = [p[2] for p in self.puntos]
        ys = [p[3] for p in self.puntos]
        for c in self.curvas:
            if c.is_vline:
                xs.append(_ffloat(c.expr))
            if c.is_hline:
                ys.append(_ffloat(c.expr))
        if not xs:
            xs = [-5.0, 5.0]
        if not ys:
            ys = [-5.0, 5.0]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        esc = max(x1 - x0, y1 - y0, 1.0)
        if x1 - x0 < 1e-9:
            x0, x1 = x0 - esc / 2, x1 + esc / 2
        if y1 - y0 < 1e-9:
            y0, y1 = y0 - esc / 2, y1 + esc / 2
        lim = 30 * esc
        gx = np.linspace(x0, x1, 600)
        gy = np.linspace(y0, y1, 600)
        yy, xx = [y0, y1], [x0, x1]
        for c in self.curvas:
            for k in range(len(c.y_branches())):
                if c.is_vline:
                    continue
                v = c.ev_y(k, gx)
                v = v[np.isfinite(v) & (np.abs(v) < lim)]
                if v.size:
                    yy += [v.min(), v.max()]
            for k in range(len(c.x_branches())):
                if c.is_hline:
                    continue
                v = c.ev_x(k, gy)
                v = v[np.isfinite(v) & (np.abs(v) < lim)]
                if v.size:
                    xx += [v.min(), v.max()]
        if self.banda_x:
            xx = [x0, x1]
        if self.banda_y:
            yy = [y0, y1]
        X0, X1, Y0, Y1 = min(xx), max(xx), min(yy), max(yy)
        px = 0.12 * (X1 - X0) + 0.05 * esc
        py = 0.12 * (Y1 - Y0) + 0.05 * esc
        return (X0 - px, X1 + px, Y0 - py, Y1 + py)

    # -- fronteras de dominio de cada rama --------------------------------
    def _fronteras(self, c, k, modo):
        """Puntos donde la rama k de la curva c empieza/termina de estar definida."""
        key = (c.idx, k, modo)
        if key in self._fr:
            return self._fr[key]
        X0, X1, Y0, Y1 = self.caja
        lo, hi = (X0, X1) if modo == "dx" else (Y0, Y1)
        ev = (lambda g: c.ev_y(k, g)) if modo == "dx" else (lambda g: c.ev_x(k, g))
        out = []
        if not (c.is_hline or c.is_vline):
            g = np.linspace(lo, hi, 1500)
            ok = np.isfinite(ev(g))
            for i in np.where(ok[:-1] != ok[1:])[0]:
                a_, b_ = g[i], g[i + 1]
                va = ok[i]
                for _ in range(60):
                    m = (a_ + b_) / 2
                    vm = np.isfinite(ev(np.array([m]))[0])
                    if vm == va:
                        a_ = m
                    else:
                        b_ = m
                v_ = a_ if va else b_
                out.append(0.0 if abs(v_) < 1e-5 else v_)
        self._fr[key] = out
        return out

    # -- raster de regiones acotadas ------------------------------------------
    def _raster(self):
        if self._enc is not None:
            return self._enc
        X0, X1, Y0, Y1 = self.caja
        n = self.NR
        mask = np.zeros((n, n), dtype=bool)  # [iy, ix]

        def cell(px, py):
            return ((px - X0) / (X1 - X0) * (n - 1),
                    (py - Y0) / (Y1 - Y0) * (n - 1))

        def trazo(px, py):
            cx, cy = cell(px, py)
            ok = np.isfinite(cx) & np.isfinite(cy)
            for k in range(len(cx) - 1):
                if not (ok[k] and ok[k + 1]):
                    continue
                dx_, dy_ = cx[k + 1] - cx[k], cy[k + 1] - cy[k]
                m = int(max(abs(dx_), abs(dy_)))
                if m > n // 2:
                    continue  # salto (asíntota): no unir
                ts = np.linspace(0, 1, m + 2)
                ix = np.rint(cx[k] + dx_ * ts).astype(int)
                iy = np.rint(cy[k] + dy_ * ts).astype(int)
                sel = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
                mask[iy[sel], ix[sel]] = True

        base_x = np.linspace(X0, X1, 2400)
        base_y = np.linspace(Y0, Y1, 2400)
        for c in self.curvas:
            if c.is_vline:
                ix = int(round(cell(_ffloat(c.expr), 0)[0]))
                if 0 <= ix < n:
                    mask[:, ix] = True
            elif c.is_hline:
                iy = int(round(cell(0, _ffloat(c.expr))[1]))
                if 0 <= iy < n:
                    mask[iy, :] = True
            elif c.y_branches():
                for k in range(len(c.y_branches())):
                    gx = np.union1d(base_x, self._fronteras(c, k, "dx"))
                    trazo(gx, c.ev_y(k, gx))
            else:
                for k in range(len(c.x_branches())):
                    gy = np.union1d(base_y, self._fronteras(c, k, "dy"))
                    trazo(c.ev_x(k, gy), gy)
        mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
        lab, _ = ndimage.label(~mask)
        borde = np.unique(np.concatenate(
            [lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
        enc = (lab > 0) & ~np.isin(lab, borde[borde > 0])
        # numerar las regiones de izquierda a derecha
        ids = np.unique(lab[enc])
        orden = sorted(ids, key=lambda q: np.where(lab == q)[1].min())
        remap = {int(q): i + 1 for i, q in enumerate(orden)}
        cara = np.zeros(lab.shape, dtype=int)
        for q, i in remap.items():
            cara[lab == q] = i
        self._enc = (cara, cell)
        return self._enc

    def encerrado(self, px, py):
        """Devuelve el nº de región acotada que contiene el punto (0 si ninguna)."""
        cara, cell = self._raster()
        n = cara.shape[0]
        cx, cy = cell(px, py)
        ix, iy = int(round(cx)), int(round(cy))
        if not (0 <= ix < n and 0 <= iy < n):
            return 0
        if cara[iy, ix]:
            return int(cara[iy, ix])
        sub = cara[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4]
        v = sub[sub > 0]
        return int(np.bincount(v).argmax()) if v.size else 0

    # -- barridos -------------------------------------------------------------
    def barrido(self, modo):
        """modo 'dx': rectángulos paralelos al eje y (ramas y=h(x));
        modo 'dy': rectángulos paralelos al eje x (ramas x=h(y))."""
        dx = modo == "dx"
        var = x if dx else y
        ramas = []  # (curva, k, expr)
        limites = []
        for c in self.curvas:
            if dx:
                if c.is_vline:
                    limites.append(c.expr)
                    continue
                if not c.y_branches():
                    return {"ok": False, "modo": modo, "msg":
                            f"No se pudo despejar y en «{c.texto}»; el barrido "
                            f"con dx no está disponible."}
                ramas += [(c, k, e) for k, e in enumerate(c.y_branches())]
            else:
                if c.is_hline:
                    limites.append(c.expr)
                    continue
                if not c.x_branches():
                    return {"ok": False, "modo": modo, "msg":
                            f"No se pudo despejar x en «{c.texto}»; el barrido "
                            f"con dy no está disponible."}
                ramas += [(c, k, e) for k, e in enumerate(c.x_branches())]
        ev = (lambda r, g: r[0].ev_y(r[1], g)) if dx else \
             (lambda r, g: r[0].ev_x(r[1], g))
        X0, X1, Y0, Y1 = self.caja
        lo, hi = (X0, X1) if dx else (Y0, Y1)

        crit = []  # (float, exacto)

        def add(v):
            try:
                f = _ffloat(v)
            except Exception:
                return
            if lo - 1e-9 <= f <= hi + 1e-9 and np.isfinite(f):
                crit.append((f, sp.sympify(v)))
        for p in self.puntos:
            add(p[0] if dx else p[1])
        for v in limites:
            add(v)
        for r in ramas:
            for v in self._fronteras(r[0], r[1], modo):
                add(snap_frontera(v))
        crit.sort(key=lambda q: q[0])
        uniq = []
        for f, e in crit:
            if not uniq or f - uniq[-1][0] > 1e-9:
                uniq.append((f, e))
        strips = []
        for (fa, ea), (fb, eb) in zip(uniq[:-1], uniq[1:]):
            vm = np.array([(fa + fb) / 2])
            vals = []
            for r in ramas:
                v = ev(r, vm)[0]
                if np.isfinite(v):
                    vals.append((v, r))
            vals.sort(key=lambda q: q[0])
            for (v1, r1), (v2, r2) in zip(vals[:-1], vals[1:]):
                if v2 - v1 < 1e-9:
                    continue
                mid = (v1 + v2) / 2
                px, py = (vm[0], mid) if dx else (mid, vm[0])
                if self.banda_x and not (self.rx[0] - 1e-9 <= px <= self.rx[1] + 1e-9):
                    continue
                if self.banda_y and not (self.ry[0] - 1e-9 <= py <= self.ry[1] + 1e-9):
                    continue
                cara = self.encerrado(px, py)
                if cara:
                    strips.append({"a": ea, "b": eb, "fa": fa, "fb": fb,
                                   "lo": r1, "hi": r2, "cara": cara})
        # fusionar tramos contiguos con los mismos "techo" y "piso"
        tramos = []
        for s in strips:
            ult = tramos[-1] if tramos else None
            if (ult and abs(ult["fb"] - s["fa"]) < 1e-9
                    and ult["lo"][0] is s["lo"][0] and ult["lo"][1] == s["lo"][1]
                    and ult["hi"][0] is s["hi"][0] and ult["hi"][1] == s["hi"][1]
                    and ult["cara"] == s["cara"]):
                ult["b"], ult["fb"] = s["b"], s["fb"]
            else:
                tramos.append(dict(s))
        tot_ex, tot_num, exacto = sp.Integer(0), 0.0, True
        for tr in tramos:
            integrando = tr["hi"][2] - tr["lo"][2]
            simp = con_limite(lambda: sp.simplify(integrando), timeout=6,
                              default=integrando)
            integrando = integrando if simp.has(sp.Piecewise) else simp
            tr["integrando"] = integrando
            res = integrar_definida(integrando, var, tr["a"], tr["b"])
            tr["res"] = res
            if res["exacto"] is None:
                exacto = False
            else:
                tot_ex += res["exacto"]
            tot_num += res["num"] if res["num"] is not None else float("nan")
        out = {"ok": True, "modo": modo, "var": var, "tramos": tramos,
               "caras": sorted({tr["cara"] for tr in tramos}),
               "num": tot_num,
               "exacto": sp.simplify(tot_ex) if exacto and tramos else None}
        if not tramos:
            out["msg"] = ("No se encontró ninguna región completamente acotada "
                          "por estas curvas. Revisá que formen un recinto cerrado.")
        return out


# ----------------------------------------------------------------------------
# 3. MÓDULOS 1-4: primitivas, PVI, TFC, integrales definidas
# ----------------------------------------------------------------------------
def _estado(clave, valor=None):
    """Guarda/recupera resultados en session_state (evita recalcular al
    cambiar de pestaña)."""
    if valor is not None:
        st.session_state[clave] = valor
    return st.session_state.get(clave)


def _error(msg):
    st.error(msg, icon="⚠️")


def _linspace_seguro(a, b, n=1800):
    return np.linspace(float(a), float(b), n)


# ---- Módulo 1 ---------------------------------------------------------------
_REGLAS_ES = {
    "URule": "sustitución (cambio de variable)",
    "PartsRule": "integración por partes",
    "CyclicPartsRule": "partes cíclica",
    "TrigSubstitutionRule": "sustitución trigonométrica",
    "PartialFractionRule": "fracciones simples",
    "RewriteRule": "acomodo algebraico / identidad",
    "PowerRule": "regla de la potencia", "ConstantRule": "integral inmediata",
    "ConstantTimesRule": "constante por función", "AddRule": "linealidad (suma)",
    "ExpRule": "exponencial inmediata", "ReciprocalRule": "1/x → ln|x|",
    "SinRule": "trigonométrica inmediata", "CosRule": "trigonométrica inmediata",
    "ArctanRule": "arco tangente inmediata", "ArcsinRule": "arco seno inmediata",
    "TrigRule": "trigonométrica inmediata", "InverseTrigRule": "inversa trigonométrica",
}


def _recorrer_reglas(regla, acc, prof=0):
    if regla is None or prof > 12:
        return
    nombre = type(regla).__name__
    if nombre.endswith("Rule"):
        acc.append(nombre)
    for v in vars(regla).values() if hasattr(regla, "__dict__") else []:
        if isinstance(v, (list, tuple)):
            for q in v:
                _recorrer_reglas(q, acc, prof + 1)
        elif hasattr(v, "__dict__"):
            _recorrer_reglas(v, acc, prof + 1)


def pistas_metodo(f):
    try:
        from sympy.integrals.manualintegrate import integral_steps
        paso = con_limite(lambda: integral_steps(f, x), timeout=6)
        acc = []
        _recorrer_reglas(paso, acc)
        etiquetas = []
        for n in acc:
            e = _REGLAS_ES.get(n)
            if e and e not in etiquetas:
                etiquetas.append(e)
        return etiquetas
    except Exception:
        return []


def calc_primitiva(txt):
    f = parse_math(txt, (x,))
    F = con_limite(lambda: sp.integrate(f, x), timeout=18)
    if F is None or F.has(sp.Integral):
        return {"ok": False, "f": f}
    Fs = con_limite(lambda: sp.simplify(F), timeout=6, default=F)
    if Fs.has(sp.Piecewise) and not F.has(sp.Piecewise):
        Fs = F
    # verificación numérica: F' = f
    g = np.linspace(0.31, 2.77, 9)
    d = num_fn(sp.diff(Fs, x), x)(g)
    v = num_fn(f, x)(g)
    ok = np.isfinite(d) & np.isfinite(v)
    verif = bool(ok.any() and np.allclose(d[ok], v[ok], rtol=1e-6, atol=1e-8))
    return {"ok": True, "f": f, "F": Fs, "verif": verif,
            "pistas": pistas_metodo(f)}


def ui_primitivas():
    st.markdown("Calculá la familia de primitivas $\\int f(x)\\,dx = F(x)+C$.")
    txt = st.text_input("f(x) =", "x*arctg(x)/(1+x^2)^2", key="m1_f")
    if st.button("Calcular primitiva", key="m1_btn", type="primary"):
        try:
            _estado("m1_res", calc_primitiva(txt))
        except ValueError as e:
            _estado("m1_res", {"error": str(e)})
    r = _estado("m1_res")
    if not r:
        return
    if "error" in r:
        _error(r["error"])
        return
    st.latex(r"\int " + L(r["f"]) + r"\,dx")
    if not r["ok"]:
        st.warning("SymPy no encontró una primitiva cerrada. Probá acomodar el "
                   "integrando o aplicar sustitución / partes manualmente.")
        return
    st.latex(r"= " + L(r["F"]) + r" + C")
    if r["verif"]:
        st.success("Verificación: derivando F(x) se recupera f(x) ✔")
    if r["pistas"]:
        st.caption("Pistas de método: " + " · ".join(r["pistas"]))
    st.caption("Una primitiva particular (C = 0):  F(x) = " + str(r["F"]))


# ---- Módulo 2 ---------------------------------------------------------------
def calc_pvi(txt, x0t, y0t, xmin, xmax):
    f = parse_math(txt, (x,))
    x0 = parse_math(x0t, ())
    y0 = parse_math(y0t, ())
    F = con_limite(lambda: sp.integrate(f, x), timeout=18)
    if F is None or F.has(sp.Integral):
        return {"ok": False, "f": f}
    F = con_limite(lambda: sp.simplify(F), timeout=6, default=F)
    Fx0 = sp.simplify(F.subs(x, x0))
    if Fx0.has(sp.zoo) or Fx0.has(sp.nan) or Fx0.has(sp.I):
        return {"ok": False, "f": f, "msg":
                "La primitiva no está definida (o no es real) en x₀."}
    C = sp.simplify(y0 - Fx0)
    Fp = F + C
    # gráfico
    xs = _linspace_seguro(xmin, xmax)
    part = num_fn(Fp, x)(xs)
    fin = part[np.isfinite(part)]
    paso = max(1.0, (np.ptp(fin) / 4 if fin.size else 1.0))
    fig = fig_base("Familia de primitivas y solución particular")
    for k in range(-4, 5):
        if k == 0:
            continue
        yk = num_fn(Fp + k * paso, x)(xs)
        fig.add_trace(go.Scatter(x=xs, y=yk, mode="lines",
                                 line=dict(color="#b0b0b0", width=1, dash="dot"),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=part, mode="lines",
                             line=dict(color="#d62728", width=3.5),
                             name="solución particular"))
    fig.add_trace(go.Scatter(
        x=[float(sp.N(x0))], y=[float(sp.N(y0))], mode="markers+text",
        marker=dict(size=12, color="#1f77b4"),
        text=["P"], textposition="top center", name="P(x₀, y₀)"))
    if fin.size:
        lo, hi = np.percentile(fin, [2, 98])
        marg = max(1.0, (hi - lo) * 0.8)
        fig.update_yaxes(range=[lo - marg, hi + marg])
    return {"ok": True, "f": f, "F": F, "x0": x0, "y0": y0, "C": C,
            "Fp": Fp, "fig": fig}


def ui_pvi():
    st.markdown("Hallá la curva $f(x)$ tal que $f'(x)$ es la función dada y "
                "pasa por $P(x_0,y_0)$.")
    txt = st.text_input("f'(x) =", "x*2^x", key="m2_f")
    c1, c2, c3 = st.columns(3)
    x0t = c1.text_input("x₀", "0", key="m2_x0")
    y0t = c2.text_input("y₀", "3/ln(2)^2", key="m2_y0")
    rng = c3.slider("Rango x", -15.0, 15.0, (-4.0, 4.0), key="m2_rng")
    if st.button("Resolver PVI", key="m2_btn", type="primary"):
        try:
            _estado("m2_res", calc_pvi(txt, x0t, y0t, rng[0], rng[1]))
        except ValueError as e:
            _estado("m2_res", {"error": str(e)})
    r = _estado("m2_res")
    if not r:
        return
    if "error" in r:
        _error(r["error"])
        return
    if not r["ok"]:
        st.warning(r.get("msg", "No se encontró una primitiva cerrada."))
        return
    st.markdown("**1) Integrar** la derivada:")
    st.latex(r"f(x)=\int " + L(r["f"]) + r"\,dx = " + L(r["F"]) + r" + C")
    st.markdown("**2) Imponer la condición** $f(x_0)=y_0$:")
    st.latex(L(r["y0"]) + r" = " + L(sp.simplify(r["F"].subs(x, r["x0"]))) +
             r" + C \;\Rightarrow\; C = " + L(r["C"]))
    st.markdown("**3) Ecuación de la curva:**")
    st.latex(r"f(x) = " + L(sp.simplify(r["Fp"])))
    mostrar_fig(r["fig"])


# ---- Módulo 3 ---------------------------------------------------------------
def calc_tfc(a_t, b_t, f_t, u_t, x0_t):
    a = parse_math(a_t, (x,))
    b = parse_math(b_t, (x,))
    f = parse_math(f_t, (t, x))
    u = parse_math(u_t, (x,))
    if x in f.free_symbols and t not in f.free_symbols:
        f = f.subs(x, t)  # el usuario usó x como variable muda
    pasos = []
    I_sym = sp.Function("I")(x)
    # estructura
    pasos.append(("Estructura",
                  r"\Phi(x)=" + (L(u) + r"\cdot " if u != 1 else "") +
                  r"\int_{" + L(a) + r"}^{" + L(b) + r"} " + L(f) + r"\,dt"))
    cte_a = not a.free_symbols
    cte_b = not b.free_symbols
    if cte_a and cte_b and x not in f.free_symbols:
        pasos.append(("Ambos límites son constantes",
                      r"\int_{a}^{b} f(t)\,dt \text{ es un número} \Rightarrow "
                      r"\Phi'(x)=0"))
        return {"pasos": pasos, "res": sp.Integer(0), "x0": None, "f": f}
    # acomodo
    if not cte_a and cte_b:
        pasos.append(("Acomodo (variable en el límite inferior)",
                      r"\int_{" + L(a) + r"}^{" + L(b) + r"} = -\int_{" + L(b) +
                      r"}^{" + L(a) + r"}"))
    elif not cte_a and not cte_b:
        pasos.append(("Acomodo (dos límites funcionales): se parte en una "
                      "constante c",
                      r"\int_{a(x)}^{b(x)} = -\int_{c}^{a(x)} + \int_{c}^{b(x)}"))
    pasos.append(("Regla (TFC + cadena)",
                  r"\frac{d}{dx}\int_{a(x)}^{b(x)} f(t)\,dt = f(b(x))\,b'(x) - "
                  r"f(a(x))\,a'(x)"))
    fb = sp.simplify(f.subs(t, b))
    fa = sp.simplify(f.subs(t, a))
    db, da = sp.diff(b, x), sp.diff(a, x)
    termino_b = sp.simplify(fb * db)
    termino_a = sp.simplify(fa * da)
    pasos.append(("Evaluar en el límite superior",
                  r"f(b)\,b' = \left(" + L(fb) + r"\right)\cdot\left(" + L(db) +
                  r"\right) = " + L(termino_b)))
    if cte_a:
        pasos.append(("Límite inferior constante", r"a'(x)=0 \Rightarrow "
                      r"\text{ese término se anula}"))
    else:
        pasos.append(("Evaluar en el límite inferior",
                      r"f(a)\,a' = \left(" + L(fa) + r"\right)\cdot\left(" +
                      L(da) + r"\right) = " + L(termino_a)))
    dI = termino_b - termino_a
    extra = None
    if x in f.free_symbols:
        dfx = sp.simplify(sp.diff(f, x))
        extra = dfx
        pasos.append(("El integrando depende de x (regla de Leibniz): se suma "
                      "la integral de la derivada parcial",
                      r"\int_{a}^{b}\frac{\partial f}{\partial x}\,dt = "
                      r"\int_{" + L(a) + r"}^{" + L(b) + r"}\left(" + L(dfx) +
                      r"\right)dt"))
    if u == 1:
        res_final = dI
        if extra is not None:
            res_final = dI + sp.Integral(extra, (t, a, b))
        else:
            res_final = sp.simplify(dI)
    else:
        integral = sp.Integral(f, (t, a, b))
        du = sp.diff(u, x)
        pasos.append(("Regla del producto (hay un factor fuera de la integral)",
                      r"\Phi' = u'(x)\int_{a}^{b} f + u(x)\,\frac{d}{dx}"
                      r"\int_{a}^{b} f"))
        inner = dI
        if extra is not None:
            inner = inner + sp.Integral(extra, (t, a, b))
        res_final = du * integral + u * inner
    pasos.append(("Resultado", r"\Phi'(x) = " + L(res_final)))
    out = {"pasos": pasos, "res": res_final, "f": f, "x0": None}
    if x0_t.strip():
        x0 = parse_math(x0_t, ())
        try:
            val = res_final.subs(x, x0)
            if val.has(sp.Integral):
                val = con_limite(lambda: val.doit(), timeout=10, default=val)
            val = sp.simplify(val)
            if val.has(sp.Integral):
                val = sp.Float(sp.N(val, 12), 12)
            out["x0"] = (x0, val)
        except Exception:
            out["x0"] = (x0, None)
    return out


def ui_tfc():
    st.markdown("Derivá $\\Phi(x)=u(x)\\int_{a(x)}^{b(x)} f(t)\\,dt$ con el TFC "
                "y la regla de la cadena.")
    c1, c2 = st.columns(2)
    a_t = c1.text_input("Límite inferior a(x)", "1", key="m3_a")
    b_t = c2.text_input("Límite superior b(x)", "e^(2*x)", key="m3_b")
    f_t = st.text_input("Integrando f(t)  (puede incluir x como parámetro)",
                        "5*(x+3)/(t^4+7)", key="m3_f")
    c3, c4 = st.columns(2)
    u_t = c3.text_input("Factor externo u(x) (opcional)", "1", key="m3_u")
    x0_t = c4.text_input("Evaluar Φ'(x₀) en x₀ (opcional)", "0", key="m3_x0")
    if st.button("Derivar paso a paso", key="m3_btn", type="primary"):
        try:
            _estado("m3_res", calc_tfc(a_t, b_t, f_t, u_t, x0_t))
        except ValueError as e:
            _estado("m3_res", {"error": str(e)})
    r = _estado("m3_res")
    if not r:
        return
    if "error" in r:
        _error(r["error"])
        return
    for i, (tit, lat) in enumerate(r["pasos"], 1):
        st.markdown(f"**{i}. {tit}**")
        st.latex(lat)
    if r.get("x0"):
        x0, val = r["x0"]
        if val is not None:
            st.latex(r"\Phi'(" + L(x0) + r") = " + L(val) +
                     (r"\approx " + N6(sp.N(val)) if not val.is_number else
                      r"\approx " + N6(sp.N(val))))


# ---- Módulo 4 ---------------------------------------------------------------
def calc_definida(f_t, a_t, b_t):
    f = parse_math(f_t, (x,))
    a = parse_math(a_t, ())
    b = parse_math(b_t, ())
    res = integrar_definida(f, x, a, b)
    G = None
    if not (a.has(sp.oo) or b.has(sp.oo) or a.has(-sp.oo)):
        G = con_limite(lambda: sp.integrate(f, x), timeout=10)
        if G is not None and G.has(sp.Integral):
            G = None
    fa, fb = float(sp.N(a)) if np.isfinite(float(sp.N(a))) else -10.0, \
        float(sp.N(b)) if np.isfinite(float(sp.N(b))) else 10.0
    lo, hi = min(fa, fb), max(fa, fb)
    pad = max(0.15 * (hi - lo), 0.5)
    xs = np.linspace(lo - pad, hi + pad, 1800)
    ys = num_fn(f, x)(xs)
    fig = fig_base("Función y área bajo la curva")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)",
                             line=dict(color="#1f3b73", width=2.5)))
    xa = np.linspace(lo, hi, 1500)
    ya = num_fn(f, x)(xa)
    pos = np.where(ya >= 0, ya, 0.0)
    neg = np.where(ya < 0, ya, 0.0)
    pos[~np.isfinite(ya)] = np.nan
    neg[~np.isfinite(ya)] = np.nan
    fig.add_trace(go.Scatter(x=xa, y=pos, mode="lines", fill="tozeroy",
                             line=dict(width=0), fillcolor="rgba(44,160,44,0.35)",
                             name="área positiva"))
    fig.add_trace(go.Scatter(x=xa, y=neg, mode="lines", fill="tozeroy",
                             line=dict(width=0), fillcolor="rgba(214,39,40,0.35)",
                             name="área negativa"))
    geo = None
    try:
        fn = num_fn(f, x)
        geo = sci_int.quad(lambda s: abs(float(fn(np.array([s]))[0])), lo, hi,
                           limit=300)[0] if np.isfinite([lo, hi]).all() else None
    except Exception:
        pass
    return {"f": f, "a": a, "b": b, "res": res, "G": G, "geo": geo, "fig": fig,
            "cambia": bool(np.isfinite(ya).any() and (ya[np.isfinite(ya)] < -1e-12).any()
                           and (ya[np.isfinite(ya)] > 1e-12).any())}


def ui_definidas():
    st.markdown("Calculá $\\int_a^b f(x)\\,dx$ con la Regla de Barrow y "
                "visualizá el área.")
    f_t = st.text_input("f(x) =", "x*sen(x)", key="m4_f")
    c1, c2 = st.columns(2)
    a_t = c1.text_input("a", "0", key="m4_a")
    b_t = c2.text_input("b", "pi", key="m4_b")
    if st.button("Calcular integral definida", key="m4_btn", type="primary"):
        try:
            _estado("m4_res", calc_definida(f_t, a_t, b_t))
        except ValueError as e:
            _estado("m4_res", {"error": str(e)})
    r = _estado("m4_res")
    if not r:
        return
    if "error" in r:
        _error(r["error"])
        return
    if r["G"] is not None:
        st.markdown("**Primitiva y Regla de Barrow**")
        st.latex(r"G(x)=" + L(r["G"]))
        st.latex(r"\int_{" + L(r["a"]) + r"}^{" + L(r["b"]) + r"} " +
                 L(r["f"]) + r"\,dx = G(" + L(r["b"]) + r") - G(" + L(r["a"]) +
                 r")")
    ex, num = fmt_valor(r["res"])
    if ex is not None:
        st.latex(r"\text{Integral} = " + ex +
                 (r" \approx " + N6(num) if num is not None else ""))
    elif num is not None:
        st.latex(r"\text{Integral} \approx " + N6(num))
        st.caption("Sin forma cerrada: se muestra el valor numérico.")
    else:
        st.warning("No se pudo evaluar la integral (¿diverge o hay una "
                   "singularidad en el intervalo?).")
    if r["cambia"] and r["geo"] is not None:
        st.info(f"f cambia de signo en [a, b]. Área geométrica "
                f"(∫|f|) ≈ {N6(r['geo'])}; la integral neta resta las "
                f"zonas rojas.")
    mostrar_fig(r["fig"])


# ----------------------------------------------------------------------------
# 4. MÓDULOS 5-7: área entre curvas, longitud de arco, sólidos de revolución
# ----------------------------------------------------------------------------
EJEMPLOS_CURVAS = {
    "Dos parábolas": ["y = x^2 - 2*x", "y = 6*x - x^2"],
    "Raíz, recta y eje x": ["y = sqrt(x)", "y = 6 - x", "y = 0"],
    "Parábola horizontal y recta": ["y^2 = 4*x", "y = 2*x - 4"],
    "Punto anguloso (techo cambia)": ["y = sqrt(x) + 1", "y = x - 1", "x = 0", "x = 2"],
    "Exponenciales": ["y = 3^x", "y = 2^(-x)", "x = -1", "x = 2"],
    "Seno y recta": ["y = sen(x)", "y = -1", "x = 0", "x = pi/2"],
    "Arco coseno (conviene dy)": ["y = arccos(x)", "y = pi/2", "y = 0", "x = 2"],
    "Con raíz cúbica": ["y = x^(1/3)", "y = x^2", "x = -1", "x = 1"],
}
EJEMPLO_VOL = {
    "y = x³, y = 1, x = 2  (eje y = -1)": ["y = x^3", "y = 1", "x = 2"],
    "y = x³, y = 1, x = 2  (eje x = -1)": ["y = x^3", "y = 1", "x = 2"],
    "y = √x + 1, y = 0, x = 4  (eje y = -1)": ["y = sqrt(x) + 1", "y = 0", "x = 4"],
    "y = ⁴√x, x = 1, y = 0  (eje y = 4)": ["y = x^(1/4)", "x = 1", "y = 0"],
    "y = x², y = 4  (eje y = 4)": ["y = x^2", "y = 4"],
}
EJE_VOL = {
    "y = x³, y = 1, x = 2  (eje y = -1)": "y = -1",
    "y = x³, y = 1, x = 2  (eje x = -1)": "x = -1",
    "y = √x + 1, y = 0, x = 4  (eje y = -1)": "y = -1",
    "y = ⁴√x, x = 1, y = 0  (eje y = 4)": "y = 4",
    "y = x², y = 4  (eje y = 4)": "y = 4",
}


def _cargar_ejemplo(prefix, ejemplos):
    lst = ejemplos[st.session_state[prefix + "_ej"]]
    st.session_state[prefix + "_n"] = len(lst)
    for i, v in enumerate(lst):
        st.session_state[f"{prefix}_c{i}"] = v


def _cargar_ejemplo_vol(prefix, ejemplos):
    _cargar_ejemplo(prefix, ejemplos)
    st.session_state["m7_eje"] = EJE_VOL[st.session_state[prefix + "_ej"]]


def _mas(prefix):
    st.session_state[prefix + "_n"] += 1


def _menos(prefix):
    n = st.session_state[prefix + "_n"]
    if n > 1:
        st.session_state[prefix + "_n"] = n - 1


def entrada_curvas(prefix, ejemplos, cb=None):
    """Lista dinámica de N ecuaciones (y = ... o x = ...)."""
    if prefix + "_n" not in st.session_state:
        primero = list(ejemplos.values())[0]
        st.session_state[prefix + "_n"] = len(primero)
        for i, v in enumerate(primero):
            st.session_state[f"{prefix}_c{i}"] = v
    st.selectbox("Ejemplos precargados", list(ejemplos.keys()),
                 key=prefix + "_ej", on_change=cb or _cargar_ejemplo,
                 args=(prefix, ejemplos))
    textos = []
    for i in range(st.session_state[prefix + "_n"]):
        if f"{prefix}_c{i}" not in st.session_state:
            st.session_state[f"{prefix}_c{i}"] = ""
        textos.append(st.text_input(f"Ecuación {i + 1}", key=f"{prefix}_c{i}",
                                    placeholder="y = x^2   ó   x = y^2 - 1"))
    c1, c2 = st.columns(2)
    c1.button("➕ Agregar ecuación", key=prefix + "_mas", on_click=_mas,
              args=(prefix,))
    c2.button("➖ Quitar última", key=prefix + "_menos", on_click=_menos,
              args=(prefix,))
    return textos


_PALETA = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
           "#8c564b", "#e377c2"]


def trazar_curvas(fig, motor, extra=1.0):
    X0, X1, Y0, Y1 = motor.caja
    wx, wy = X1 - X0, Y1 - Y0
    gx = np.linspace(X0 - extra * wx, X1 + extra * wx, 3000)
    gy = np.linspace(Y0 - extra * wy, Y1 + extra * wy, 3000)
    for i, c in enumerate(motor.curvas):
        col = _PALETA[i % len(_PALETA)]
        nombre = c.texto
        primero = True
        if c.is_vline:
            v = float(sp.N(c.expr))
            fig.add_trace(go.Scatter(x=[v, v], y=gy[[0, -1]], mode="lines",
                                     line=dict(color=col, width=2.2), name=nombre))
            continue
        if c.y_branches():
            for k in range(len(c.y_branches())):
                fig.add_trace(go.Scatter(
                    x=gx, y=c.ev_y(k, gx), mode="lines", connectgaps=False,
                    line=dict(color=col, width=2.2), name=nombre,
                    legendgroup=f"c{i}", showlegend=primero))
                primero = False
        else:
            for k in range(len(c.x_branches())):
                fig.add_trace(go.Scatter(
                    x=c.ev_x(k, gy), y=gy, mode="lines", connectgaps=False,
                    line=dict(color=col, width=2.2), name=nombre,
                    legendgroup=f"c{i}", showlegend=primero))
                primero = False
    pad = 0.08
    fig.update_xaxes(range=[X0 - pad * wx, X1 + pad * wx])
    fig.update_yaxes(range=[Y0 - pad * wy, Y1 + pad * wy])


def marcar_puntos(fig, motor):
    if not motor.puntos:
        return
    fig.add_trace(go.Scatter(
        x=[p[2] for p in motor.puntos], y=[p[3] for p in motor.puntos],
        mode="markers", name="intersecciones",
        marker=dict(size=10, color="black", symbol="circle-open", line=dict(width=2)),
        hovertext=[f"({p[0]}, {p[1]}) ≈ ({p[2]:.4g}, {p[3]:.4g})"
                   for p in motor.puntos], hoverinfo="text"))


def sombrear(fig, motor, r, sel, rect_dif=True):
    """Sombrea las regiones elegidas y dibuja un rectángulo diferencial por tramo."""
    dx = r["modo"] == "dx"
    vistas = set()
    for n, tr in enumerate(r["tramos"]):
        if tr["cara"] not in sel:
            continue
        s = np.linspace(tr["fa"], tr["fb"], 300)
        hi = (tr["hi"][0].ev_y(tr["hi"][1], s) if dx
              else tr["hi"][0].ev_x(tr["hi"][1], s))
        lo = (tr["lo"][0].ev_y(tr["lo"][1], s) if dx
              else tr["lo"][0].ev_x(tr["lo"][1], s))
        ok = np.isfinite(hi) & np.isfinite(lo)
        s, hi, lo = s[ok], hi[ok], lo[ok]
        if s.size < 2:
            continue
        px = np.concatenate([s, s[::-1]]) if dx else np.concatenate([hi, lo[::-1]])
        py = np.concatenate([hi, lo[::-1]]) if dx else np.concatenate([s, s[::-1]])
        fig.add_trace(go.Scatter(
            x=px, y=py, fill="toself", mode="lines", line=dict(width=0),
            fillcolor="rgba(255,127,14,0.30)", name=f"región R{tr['cara']}",
            legendgroup=f"R{tr['cara']}", showlegend=(tr["cara"] not in vistas),
            hoverinfo="skip"))
        vistas.add(tr["cara"])
        if rect_dif:
            m = len(s) // 2
            w = (s[-1] - s[0]) / 40.0
            if dx:
                rx = [s[m] - w / 2, s[m] + w / 2, s[m] + w / 2, s[m] - w / 2, s[m] - w / 2]
                ry = [lo[m], lo[m], hi[m], hi[m], lo[m]]
            else:
                ry = [s[m] - w / 2, s[m] + w / 2, s[m] + w / 2, s[m] - w / 2, s[m] - w / 2]
                rx = [lo[m], lo[m], hi[m], hi[m], lo[m]]
            fig.add_trace(go.Scatter(
                x=rx, y=ry, mode="lines", fill="toself",
                line=dict(color="#7f3c00", width=1.5, dash="dash"),
                fillcolor="rgba(127,60,0,0.35)", showlegend=False,
                hoverinfo="skip"))


def latex_tramos(r, sel, plantilla):
    """Arma  A = ∫[(arriba) − (abajo)]dx + ∫ ...  para las regiones elegidas."""
    var = "dx" if r["modo"] == "dx" else "dy"
    partes = []
    for tr in r["tramos"]:
        if tr["cara"] not in sel:
            continue
        hi, lo = tr["hi"][2], tr["lo"][2]
        partes.append(r"\int_{" + L(tr["a"]) + r"}^{" + L(tr["b"]) + r"}" +
                      plantilla(L(hi), L(lo)) + r"\," + var)
    return " + ".join(partes)


def suma_area(r, sel):
    ex, num, exacto = sp.Integer(0), 0.0, True
    for tr in r["tramos"]:
        if tr["cara"] not in sel:
            continue
        res = tr["res"]
        if res["exacto"] is None:
            exacto = False
        else:
            ex += res["exacto"]
        num += res["num"] if res["num"] is not None else float("nan")
    ex = limpiar_exacto(sp.simplify(ex)) if exacto else None
    return ex, num


def mostrar_barrido(r, sel, etiquetas):
    """Muestra el planteo de área de un barrido (dx o dy)."""
    tit, arriba, abajo = etiquetas
    if not r["ok"]:
        st.warning(r["msg"])
        return None
    if not r["tramos"]:
        st.warning(r.get("msg", "Sin regiones acotadas."))
        return None
    st.markdown(f"**{tit}**")
    plantilla = lambda a, b: r"\left[\left(" + a + r"\right)-\left(" + b + r"\right)\right]"
    lat = latex_tramos(r, sel, plantilla)
    if not lat:
        st.info("Elegí al menos una región.")
        return None
    ntr = sum(1 for tr in r["tramos"] if tr["cara"] in sel)
    if ntr > 1:
        st.caption(f"{arriba}/{abajo}: la región se parte en {ntr} integrales "
                   f"porque cambia el «techo» o el «piso» (puntos angulosos).")
    st.latex(r"A = " + lat)
    ex, num = suma_area(r, sel)
    st.latex(r"A = " + (L(ex) + r" \approx " if ex is not None else r"\approx ") + N6(num))
    return num


# ---- Módulo 5 ---------------------------------------------------------------
def ui_area():
    st.markdown("Ingresá **N ecuaciones** (`y = ...` o `x = ...`). El motor "
                "detecta las intersecciones y las regiones cerradas, y calcula "
                "el área con rectángulos paralelos al eje **y** (dx) y al eje "
                "**x** (dy).")
    textos = entrada_curvas("m5", EJEMPLOS_CURVAS)
    ventana = st.number_input("Ventana de búsqueda de intersecciones (±)",
                              1.0, 200.0, 30.0, key="m5_win")
    if st.button("Calcular área", key="m5_btn", type="primary"):
        try:
            with st.spinner("Analizando curvas…"):
                motor = Motor(textos, ventana)
                st.session_state["m5_res"] = {
                    "motor": motor, "dx": motor.barrido("dx"),
                    "dy": motor.barrido("dy")}
        except ValueError as e:
            st.session_state["m5_res"] = {"error": str(e)}
    r = st.session_state.get("m5_res")
    if not r:
        return
    if "error" in r:
        _error(r["error"])
        return
    motor, rdx, rdy = r["motor"], r["dx"], r["dy"]
    base = rdx if rdx.get("ok") and rdx["tramos"] else rdy
    caras = base.get("caras", []) if base.get("ok") else []
    if len(caras) > 1:
        sel = st.multiselect(
            "Regiones cerradas detectadas (R1, R2, … de izquierda a derecha)",
            caras, default=caras, format_func=lambda q: f"R{q}",
            key="m5_sel_" + "_".join(str(q) for q in caras))
    else:
        sel = caras
    fig = fig_base("Curvas, intersecciones y región", alto=520)
    trazar_curvas(fig, motor)
    if base.get("ok") and base["tramos"]:
        sombrear(fig, motor, base, sel)
    marcar_puntos(fig, motor)
    mostrar_fig(fig)
    if motor.puntos:
        with st.expander("Puntos de intersección", expanded=False):
            for i, p in enumerate(motor.puntos, 1):
                st.latex(rf"P_{{{i}}}=\left({L(p[0])},\,{L(p[1])}\right)"
                         rf"\approx({p[2]:.5g},\,{p[3]:.5g})")
    a1 = mostrar_barrido(rdx, sel, ("Rectángulos paralelos al eje Y (dx):",
                                    "Arriba", "Abajo"))
    a2 = mostrar_barrido(rdy, sel, ("Rectángulos paralelos al eje X (dy):",
                                    "Derecha", "Izquierda"))
    if a1 is not None and a2 is not None:
        if abs(a1 - a2) < 1e-6 * max(1, abs(a1)):
            st.success("Comprobación: ambos barridos dan la misma área ✔")
        else:
            st.error("Los dos barridos no coinciden; revisá las ecuaciones.")


# ---- Módulo 6 ---------------------------------------------------------------
def calc_longitud(tipo, f_t, a_t, b_t):
    var = x if tipo.startswith("y") else y
    f = parse_math(f_t, (var,))
    a, b = parse_math(a_t, ()), parse_math(b_t, ())
    d = sp.diff(f, var)
    rad = sp.simplify(1 + d ** 2)
    integrando = sp.sqrt(rad)
    sq = con_limite(lambda: sp.simplify(integrando), timeout=6, default=integrando)
    integrando = sq if not sq.has(sp.Piecewise) else integrando
    res = integrar_definida(integrando, var, a, b)
    fa, fb = float(sp.N(a)), float(sp.N(b))
    lo, hi = min(fa, fb), max(fa, fb)
    pad = max(0.4 * (hi - lo), 0.5)
    s = np.linspace(lo - pad, hi + pad, 2000)
    sa = np.linspace(lo, hi, 1000)
    fn = num_fn(f, var)
    fig = fig_base("Curva completa y arco medido", alto=470)
    if var == x:
        fig.add_trace(go.Scatter(x=s, y=fn(s), mode="lines", name="curva",
                                 line=dict(color="#9aa5b1", width=2)))
        fig.add_trace(go.Scatter(x=sa, y=fn(sa), mode="lines", name="arco medido",
                                 line=dict(color="#d62728", width=5)))
        ex_, ey_ = [lo, hi], [float(fn(np.array([lo]))[0]), float(fn(np.array([hi]))[0])]
    else:
        fig.add_trace(go.Scatter(x=fn(s), y=s, mode="lines", name="curva",
                                 line=dict(color="#9aa5b1", width=2)))
        fig.add_trace(go.Scatter(x=fn(sa), y=sa, mode="lines", name="arco medido",
                                 line=dict(color="#d62728", width=5)))
        ex_, ey_ = [float(fn(np.array([lo]))[0]), float(fn(np.array([hi]))[0])], [lo, hi]
    fig.add_trace(go.Scatter(x=ex_, y=ey_, mode="markers+text", text=["A", "B"],
                             textposition="top center", name="extremos",
                             marker=dict(size=11, color="black")))
    return {"f": f, "var": var, "d": d, "integrando": integrando, "res": res,
            "a": a, "b": b, "fig": fig}


def ui_longitud():
    st.markdown("Longitud del arco de una curva en un intervalo.")
    tipo = st.radio("Tipo de curva", ["y = f(x)", "x = g(y)"], horizontal=True,
                    key="m6_tipo")
    f_t = st.text_input("Función", "x^(3/2) + 10", key="m6_f")
    c1, c2 = st.columns(2)
    a_t = c1.text_input("a", "4/3", key="m6_a")
    b_t = c2.text_input("b", "32/9", key="m6_b")
    if st.button("Calcular longitud", key="m6_btn", type="primary"):
        try:
            _estado("m6_res", calc_longitud(tipo, f_t, a_t, b_t))
        except ValueError as e:
            _estado("m6_res", {"error": str(e)})
    r = _estado("m6_res")
    if not r:
        return
    if "error" in r:
        _error(r["error"])
        return
    v = "x" if r["var"] == x else "y"
    w = "y" if v == "x" else "x"
    st.latex(r"L=\int_{a}^{b}\sqrt{1+\left[" + w + r"'(" + v + r")\right]^2}\,d" + v)
    st.latex(w + r"'(" + v + r") = " + L(r["d"]))
    st.latex(r"L=\int_{" + L(r["a"]) + r"}^{" + L(r["b"]) + r"}" +
             L(r["integrando"]) + r"\,d" + v)
    ex, num = fmt_valor(r["res"])
    if ex is not None:
        st.latex(r"L = " + ex + (r" \approx " + N6(num) if num is not None else ""))
    elif num is not None:
        st.latex(r"L \approx " + N6(num))
        st.caption("Sin primitiva elemental: valor numérico.")
    else:
        st.warning("No se pudo calcular la longitud en ese intervalo.")
    mostrar_fig(r["fig"])


# ---- Módulo 7 ---------------------------------------------------------------
def parse_eje(txt):
    s = txt.strip().lower().replace(" ", "")
    if s in ("ejex", "eje_x", "y=0"):
        return ("h", sp.Integer(0))
    if s in ("ejey", "eje_y", "x=0"):
        return ("v", sp.Integer(0))
    if s.count("=") != 1:
        raise ValueError("El eje de giro debe escribirse como  y = c  (horizontal) "
                         "o  x = c  (vertical).")
    izq, der = s.split("=")
    if izq == "y":
        return ("h", parse_math(der, ()))
    if izq == "x":
        return ("v", parse_math(der, ()))
    raise ValueError("El eje de giro debe escribirse como  y = c  o  x = c.")


def calc_volumen(textos, eje_t, ventana):
    eje = parse_eje(eje_t)
    motor = Motor(textos, ventana)
    modo = "dx" if eje[0] == "h" else "dy"
    r = motor.barrido(modo)
    return {"motor": motor, "r": r, "eje": eje}


def tramo_volumen(tr, var, c):
    """Devuelve (Rext, Rint, integrando, mensaje) para un tramo."""
    cf = float(sp.N(c))
    s = np.linspace(tr["fa"], tr["fb"], 60)[1:-1]
    dh = num_fn(tr["hi"][2], var)(s) - cf
    dl = num_fn(tr["lo"][2], var)(s) - cf
    ok = np.isfinite(dh) & np.isfinite(dl)
    dh, dl = dh[ok], dl[ok]
    if dh.size == 0:
        return None, None, None, "sin datos"
    if np.all(dh >= -1e-9) and np.all(dl >= -1e-9):
        Rext, Rint = tr["hi"][2] - c, tr["lo"][2] - c
    elif np.all(dh <= 1e-9) and np.all(dl <= 1e-9):
        Rext, Rint = c - tr["lo"][2], c - tr["hi"][2]
    else:
        return None, None, None, ("El eje de giro atraviesa el interior de la "
                                  "región: no se puede aplicar la fórmula.")
    return Rext, Rint, sp.simplify(Rext ** 2 - Rint ** 2), ""


def fig_3d(motor, r, eje, sel, ang):
    tipo, c = eje
    cf = float(sp.N(c))
    fig = go.Figure()
    th = np.linspace(0, np.deg2rad(ang), 70)
    dx = r["modo"] == "dx"
    xmin = ymin = np.inf
    xmax = ymax = -np.inf

    def revolver(s, R):
        if tipo == "h":
            X = s[:, None] * np.ones_like(th)[None, :]
            Yv = cf + R[:, None] * np.cos(th)[None, :]
            Z = R[:, None] * np.sin(th)[None, :]
            return X, Yv, Z
        X = cf + R[:, None] * np.cos(th)[None, :]
        Yv = s[:, None] * np.ones_like(th)[None, :]
        Z = R[:, None] * np.sin(th)[None, :]
        return X, Yv, Z

    def superficie(X, Yv, Z, col, nombre, op=0.85):
        fig.add_trace(go.Surface(
            x=X, y=Yv, z=Z, showscale=False, opacity=op, name=nombre,
            colorscale=[[0, col], [1, col]], showlegend=False, hoverinfo="skip"))

    for tr in r["tramos"]:
        if tr["cara"] not in sel:
            continue
        s = np.linspace(tr["fa"], tr["fb"], 140)
        vh = (tr["hi"][0].ev_y(tr["hi"][1], s) if dx else tr["hi"][0].ev_x(tr["hi"][1], s))
        vl = (tr["lo"][0].ev_y(tr["lo"][1], s) if dx else tr["lo"][0].ev_x(tr["lo"][1], s))
        ok = np.isfinite(vh) & np.isfinite(vl)
        s, vh, vl = s[ok], vh[ok], vl[ok]
        if s.size < 3:
            continue
        for vals, col in ((vh, "#1f77b4"), (vl, "#ff7f0e")):
            superficie(*revolver(s, vals - cf), col, "frontera")
        for idx in (0, -1):  # tapas (arandelas) en los extremos del tramo
            rr = np.linspace(vl[idx] - cf, vh[idx] - cf, 10)
            if abs(rr[0] - rr[-1]) < 1e-12:
                continue
            sv = np.full(rr.shape, s[idx])
            superficie(*revolver(sv, rr), "#2ca02c", "tapa", op=0.7)
    # eje de giro
    X0, X1, Y0, Y1 = motor.caja
    if tipo == "h":
        ex, ey = [X0, X1], [cf, cf]
    else:
        ex, ey = [cf, cf], [Y0, Y1]
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=[0, 0], mode="lines",
                               line=dict(color="black", width=6), name="eje de giro"))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0),
                      scene=dict(aspectmode="data", xaxis_title="x",
                                 yaxis_title="y", zaxis_title="z"),
                      title=dict(text="Sólido de revolución (3D interactivo)",
                                 font=dict(size=14)))
    return fig


def ui_volumen():
    modo_calc = st.radio("Calcular", ["Volumen de un sólido", "Área de superficie"],
                         horizontal=True, key="m7_modo")
    if modo_calc.startswith("Volumen"):
        _ui_volumen()
    else:
        _ui_superficie()


def _ui_volumen():
    st.markdown("Región plana limitada por las curvas dadas, girada alrededor "
                "de un eje. **Fórmula general:** radio = |curva − eje|, "
                "$V=\\pi\\int\\left([R_{ext}]^2-[R_{int}]^2\\right)$.")
    if "m7_eje" not in st.session_state:
        st.session_state["m7_eje"] = "y = -1"
    textos = entrada_curvas("m7", EJEMPLO_VOL, cb=_cargar_ejemplo_vol)
    c1, c2 = st.columns(2)
    eje_t = c1.text_input("Eje de giro (y = c  ó  x = c)", key="m7_eje")
    ang = c2.slider("Ángulo de barrido 3D (°) — menos de 360° deja ver el hueco",
                    90, 360, 300, 15, key="m7_ang")
    ventana = st.number_input("Ventana de búsqueda de intersecciones (±)",
                              1.0, 200.0, 30.0, key="m7_win")
    if st.button("Calcular volumen", key="m7_btn", type="primary"):
        try:
            with st.spinner("Calculando el sólido…"):
                st.session_state["m7_res"] = calc_volumen(textos, eje_t, ventana)
        except ValueError as e:
            st.session_state["m7_res"] = {"error": str(e)}
    res = st.session_state.get("m7_res")
    if not res:
        return
    if "error" in res:
        _error(res["error"])
        return
    motor, r, eje = res["motor"], res["r"], res["eje"]
    if not r["ok"]:
        st.warning(r["msg"])
        return
    if not r["tramos"]:
        st.warning(r.get("msg", "No hay región acotada."))
        return
    caras = r["caras"]
    sel = st.multiselect("Regiones a girar", caras, default=caras,
                         format_func=lambda q: f"R{q}",
                         key="m7_sel_" + "_".join(str(q) for q in caras)) \
        if len(caras) > 1 else caras
    tipo, c = eje
    var = r["var"]
    st.markdown("**Planteo** — " + (
        "eje horizontal ⇒ el corte es vertical ⇒ diferencial **dx**"
        if tipo == "h" else
        "eje vertical ⇒ el corte es horizontal ⇒ diferencial **dy** (se usan "
        "las funciones inversas, x despejada)"))
    total_ex, total_num, exacto, err = sp.Integer(0), 0.0, True, False
    for n, tr in enumerate([q for q in r["tramos"] if q["cara"] in sel], 1):
        Rext, Rint, integ, msg = tramo_volumen(tr, var, c)
        if msg:
            st.error(f"Tramo {n}: {msg}")
            err = True
            continue
        st.latex(rf"\text{{Tramo {n}: }}R_{{ext}}={L(sp.simplify(Rext))},\;"
                 rf"R_{{int}}={L(sp.simplify(Rint))}")
        st.latex(r"V_{" + str(n) + r"}=\pi\int_{" + L(tr["a"]) + r"}^{" + L(tr["b"]) +
                 r"}\left(\left[" + L(sp.simplify(Rext)) + r"\right]^2-\left[" +
                 L(sp.simplify(Rint)) + r"\right]^2\right)d" + ("x" if tipo == "h" else "y"))
        resv = integrar_definida(integ, var, tr["a"], tr["b"])
        if resv["exacto"] is None:
            exacto = False
        else:
            total_ex += sp.pi * resv["exacto"]
        total_num += math_pi() * (resv["num"] if resv["num"] is not None else float("nan"))
    if not err:
        st.latex(r"V = " + (L(limpiar_exacto(sp.simplify(total_ex))) + r"\approx " if exacto else r"\approx ") +
                 N6(total_num) + r"\ \text{u}^3")
    fig = fig_base("Región plana y eje de giro", alto=460)
    trazar_curvas(fig, motor)
    sombrear(fig, motor, r, sel)
    X0, X1, Y0, Y1 = motor.caja
    cf = float(sp.N(c))
    if tipo == "h":
        fig.add_trace(go.Scatter(x=[X0 - 2 * (X1 - X0), X1 + 2 * (X1 - X0)], y=[cf, cf],
                                 mode="lines", name="eje de giro",
                                 line=dict(color="black", width=3, dash="dashdot")))
    else:
        fig.add_trace(go.Scatter(x=[cf, cf], y=[Y0 - 2 * (Y1 - Y0), Y1 + 2 * (Y1 - Y0)],
                                 mode="lines", name="eje de giro",
                                 line=dict(color="black", width=3, dash="dashdot")))
    marcar_puntos(fig, motor)
    mostrar_fig(fig)
    if not err:
        mostrar_fig(fig_3d(motor, r, eje, sel, ang))


def math_pi():
    return float(np.pi)


def calc_superficie(tipo_c, f_t, a_t, b_t, eje_t):
    eje = parse_eje(eje_t)
    var = x if tipo_c.startswith("y") else y
    f = parse_math(f_t, (var,))
    a, b = parse_math(a_t, ()), parse_math(b_t, ())
    d = sp.diff(f, var)
    ds = sp.sqrt(sp.simplify(1 + d ** 2))
    px_, py_ = (var, f) if var == x else (f, var)
    radio = (py_ if eje[0] == "h" else px_) - eje[1]
    am, bm = float(sp.N(a)), float(sp.N(b))
    s = np.linspace(am, bm, 200)[1:-1]
    rv = num_fn(radio, var)(s)
    rv = rv[np.isfinite(rv)]
    aviso = ""
    if rv.size and np.all(rv >= -1e-9):
        r_eff = radio
    elif rv.size and np.all(rv <= 1e-9):
        r_eff = -radio
    else:
        r_eff = sp.Abs(radio)
        aviso = "La curva cruza el eje de giro: se usa |radio|."
    integrando = sp.simplify(r_eff * ds)
    res = integrar_definida(integrando, var, a, b)
    # 3D
    g = np.linspace(am, bm, 160)
    fn = num_fn(f, var)
    vals = fn(g)
    xs_, ys_ = (g, vals) if var == x else (vals, g)
    th = np.linspace(0, 2 * np.pi, 80)
    cf = float(sp.N(eje[1]))
    if eje[0] == "h":
        R = ys_ - cf
        X = xs_[:, None] * np.ones_like(th)[None, :]
        Yv = cf + R[:, None] * np.cos(th)[None, :]
        Z = R[:, None] * np.sin(th)[None, :]
    else:
        R = xs_ - cf
        X = cf + R[:, None] * np.cos(th)[None, :]
        Yv = ys_[:, None] * np.ones_like(th)[None, :]
        Z = R[:, None] * np.sin(th)[None, :]
    fig = go.Figure(go.Surface(x=X, y=Yv, z=Z, showscale=False, opacity=0.9,
                               colorscale="Viridis"))
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0),
                      scene=dict(aspectmode="data"),
                      title=dict(text="Superficie de revolución (3D)",
                                 font=dict(size=14)))
    return {"var": var, "ds": ds, "radio": radio, "r_eff": r_eff,
            "integrando": integrando, "res": res, "a": a, "b": b,
            "aviso": aviso, "fig": fig, "tipo": eje[0]}


def _ui_superficie():
    st.markdown("$S=2\\pi\\int r\\,ds$, con $r$ = distancia de la curva al eje de "
                "giro y $ds=\\sqrt{1+(f')^2}\\,dx$.")
    tipo_c = st.radio("Tipo de curva", ["y = f(x)", "x = g(y)"], horizontal=True,
                      key="m7s_tipo")
    f_t = st.text_input("Función", "4*sqrt(x)", key="m7s_f")
    c1, c2, c3 = st.columns(3)
    a_t = c1.text_input("a", "5", key="m7s_a")
    b_t = c2.text_input("b", "21", key="m7s_b")
    eje_t = c3.text_input("Eje (y = c ó x = c)", "y = 0", key="m7s_eje")
    if st.button("Calcular superficie", key="m7s_btn", type="primary"):
        try:
            _estado("m7s_res", calc_superficie(tipo_c, f_t, a_t, b_t, eje_t))
        except ValueError as e:
            _estado("m7s_res", {"error": str(e)})
    r = _estado("m7s_res")
    if not r:
        return
    if "error" in r:
        _error(r["error"])
        return
    v = "x" if r["var"] == x else "y"
    st.latex(r"S=2\pi\int_{" + L(r["a"]) + r"}^{" + L(r["b"]) + r"}\left|" +
             L(r["radio"]) + r"\right|\sqrt{1+(\ldots)^2}\,d" + v)
    st.latex(r"S=2\pi\int_{" + L(r["a"]) + r"}^{" + L(r["b"]) + r"}" +
             L(r["integrando"]) + r"\,d" + v)
    if r["aviso"]:
        st.info(r["aviso"])
    ex, num = fmt_valor(r["res"])
    if ex is not None:
        st.latex(r"S = 2\pi\left(" + ex + r"\right) = " + L(sp.simplify(2 * sp.pi * r["res"]["exacto"])) +
                 r" \approx " + N6(2 * np.pi * num))
    elif num is not None:
        st.latex(r"S \approx " + N6(2 * np.pi * num))
    else:
        st.warning("No se pudo evaluar la integral.")
    mostrar_fig(r["fig"])


# ----------------------------------------------------------------------------
# 5. MÓDULOS 8-9: valor promedio y límites
# ----------------------------------------------------------------------------
def calc_promedio(f_t, a_t, b_t):
    f = parse_math(f_t, (x,))
    a, b = parse_math(a_t, ()), parse_math(b_t, ())
    af, bf = float(sp.N(a)), float(sp.N(b))
    if not (np.isfinite(af) and np.isfinite(bf)) or bf <= af:
        raise ValueError("Se necesita un intervalo finito [a, b] con a < b.")
    res = integrar_definida(f, x, a, b)
    if res["num"] is None:
        return {"f": f, "a": a, "b": b, "res": res, "vp": None}
    vp_ex = sp.simplify(res["exacto"] / (b - a)) if res["exacto"] is not None else None
    vp = res["num"] / (bf - af)
    fn = num_fn(f, x)
    cs = raices_mixtas(lambda g: fn(g) - vp,
                       (f - vp_ex) if vp_ex is not None else None, x, af, bf)
    cs = [c for c in cs if af - 1e-9 <= float(sp.N(c)) <= bf + 1e-9]
    cs = [c if sp.count_ops(c) <= 30 else sp.Float(sp.N(c, 12), 12) for c in cs]
    pad = max(0.2 * (bf - af), 0.3)
    xs = np.linspace(af - pad, bf + pad, 1800)
    xa = np.linspace(af, bf, 1200)
    fig = fig_base("Área bajo f y rectángulo equivalente de altura f_prom")
    fig.add_trace(go.Scatter(x=xs, y=fn(xs), mode="lines", name="f(x)",
                             line=dict(color="#1f3b73", width=2.5)))
    fig.add_trace(go.Scatter(x=xa, y=fn(xa), mode="lines", fill="tozeroy",
                             line=dict(width=0), name="área bajo f",
                             fillcolor="rgba(31,119,180,0.30)"))
    fig.add_trace(go.Scatter(
        x=[af, bf, bf, af, af], y=[0, 0, vp, vp, 0], mode="lines", fill="toself",
        name="rectángulo equivalente", line=dict(color="#d62728", width=2, dash="dash"),
        fillcolor="rgba(214,39,40,0.18)"))
    fig.add_trace(go.Scatter(x=[af - pad, bf + pad], y=[vp, vp], mode="lines",
                             line=dict(color="#d62728", width=1.2), showlegend=False,
                             hoverinfo="skip"))
    if cs:
        fig.add_trace(go.Scatter(
            x=[float(sp.N(c)) for c in cs], y=[vp] * len(cs), mode="markers+text",
            text=["c"] * len(cs), textposition="top center", name="f(c) = f_prom",
            marker=dict(size=12, color="black")))
    return {"f": f, "a": a, "b": b, "res": res, "vp": vp, "vp_ex": vp_ex,
            "cs": cs, "fig": fig}


def ui_promedio():
    st.markdown("$f_{prom}=\\dfrac{1}{b-a}\\int_a^b f(x)\\,dx$ y el "
                "punto $c$ del Teorema del Valor Medio.")
    f_t = st.text_input("f(x) =", "x/(3*x+2)^(1/3)", key="m8_f")
    c1, c2 = st.columns(2)
    a_t = c1.text_input("a", "-1/3", key="m8_a")
    b_t = c2.text_input("b", "2", key="m8_b")
    if st.button("Calcular valor promedio", key="m8_btn", type="primary"):
        try:
            _estado("m8_res", calc_promedio(f_t, a_t, b_t))
        except ValueError as e:
            _estado("m8_res", {"error": str(e)})
    r = _estado("m8_res")
    if not r:
        return
    if "error" in r:
        _error(r["error"])
        return
    if r["vp"] is None:
        st.warning("No se pudo evaluar la integral en ese intervalo.")
        return
    ex, num = fmt_valor(r["res"])
    st.latex(r"\int_{" + L(r["a"]) + r"}^{" + L(r["b"]) + r"}" + L(r["f"]) +
             r"\,dx = " + (ex if ex is not None else r"\approx " + N6(num)))
    st.latex(r"f_{prom}=\frac{1}{" + L(sp.simplify(r["b"] - r["a"])) + r"}\cdot " +
             (r"\left(" + ex + r"\right)" if ex is not None else N6(num)) +
             r"=" + (L(r["vp_ex"]) + r"\approx " if r["vp_ex"] is not None else
                     r"\approx ") + N6(r["vp"]))
    if r["cs"]:
        st.markdown("**Teorema del Valor Medio:** existe $c\\in[a,b]$ con "
                    "$f(c)=f_{prom}$:")
        for c in r["cs"]:
            st.latex(r"c = " + L(c) + r"\approx " + N6(sp.N(c)))
    mostrar_fig(r["fig"])


# ---- Límites ------------------------------------------------------------------
_DIRS = {"Bilateral": "+-", "Por la derecha (x → c⁺)": "+",
         "Por la izquierda (x → c⁻)": "-"}


def _lim(e, c, d):
    def trabajo():
        if c in (sp.oo, -sp.oo):
            return sp.limit(e, x, c)
        return sp.limit(e, x, c, d)
    v = con_limite(trabajo, timeout=8)
    if v is None or v.has(sp.Limit):
        return None
    return v


def _inf(v):
    return v is not None and v in (sp.oo, -sp.oo, sp.zoo)


def tipo_forma(e, c, d):
    e = sp.sympify(e)
    b, ex = e.as_base_exp()
    if ex.has(x) and b != sp.E:
        lb, le = _lim(b, c, d), _lim(ex, c, d)
        if lb is not None and le is not None:
            if lb == 1 and _inf(le):
                return "1^∞"
            if lb == 0 and le == 0:
                return "0^0"
            if lb == sp.oo and le == 0:
                return "∞^0"
    num, den = sp.fraction(sp.together(e))
    if den.has(x):
        ln_, ld = _lim(num, c, d), _lim(den, c, d)
        if ln_ == 0 and ld == 0:
            return "0/0"
        if _inf(ln_) and _inf(ld):
            return "∞/∞"
    if e.is_Mul:
        ls = [_lim(q, c, d) for q in e.args]
        if any(q == 0 for q in ls) and any(_inf(q) for q in ls):
            return "0·∞"
    if e.is_Add:
        ls = [_lim(q, c, d) for q in e.args]
        if sp.oo in ls and -sp.oo in ls:
            return "∞−∞"
    return None


def _lat_lim(expr, c, d):
    arrow = {"+": r"x\to " + L(c) + "^{+}", "-": r"x\to " + L(c) + "^{-}",
             "+-": r"x\to " + L(c)}[d]
    return r"\lim_{" + arrow + r"}\;" + L(expr)


def resolver_cociente(n, dd, c, d, pasos):
    """L'Hôpital sobre n/dd (0/0 o ∞/∞) con hasta 5 derivaciones."""
    for k in range(5):
        n1 = sp.simplify(sp.diff(n, x))
        d1 = sp.simplify(sp.diff(dd, x))
        pasos.append((f"L'Hôpital ({k + 1}.ª aplicación): derivo numerador y "
                      f"denominador por separado",
                      r"\lim\frac{(" + L(n) + r")'}{(" + L(dd) + r")'} = \lim\frac{" +
                      L(n1) + "}{" + L(d1) + "}"))
        q = sp.simplify(n1 / d1)
        f2 = tipo_forma(q, c, d)
        if f2 is None:
            val = _lim(q, c, d)
            pasos.append(("Ya no hay indeterminación: evalúo", _lat_lim(q, c, d) +
                          (r" = " + L(val) if val is not None else "")))
            return val
        if f2 in ("0/0", "∞/∞") and q.is_rational_function(x):
            val = _lim(q, c, d)
            pasos.append(("Cociente de polinomios: se simplifica / se divide por la "
                          "mayor potencia de x",
                          _lat_lim(q, c, d) + (r" = " + L(val) if val is not None else "")))
            return val
        if f2 not in ("0/0", "∞/∞"):
            return None  # forma distinta: se recurre a la verificación simbólica
        n, dd = n1, d1
        pasos.append((f"Sigue siendo {f2}: se repite", ""))
    return None


def resolver_limite(e, c, d, pasos, prof=0):
    e = sp.sympify(e)
    forma = tipo_forma(e, c, d)
    if forma is None:
        val = _lim(e, c, d)
        pasos.append(("Evaluación directa (sin indeterminación)",
                      _lat_lim(e, c, d) + (r" = " + L(val) if val is not None else "")))
        return val
    pasos.append((f"Forma indeterminada detectada: {forma}", _lat_lim(e, c, d)))
    if forma in ("0/0", "∞/∞"):
        n, dd = sp.fraction(sp.together(e))
        for nombre, fn in (
                ("Salvataje algebraico: factorizo y simplifico",
                 lambda q: sp.cancel(sp.factor(q))),
                ("Salvataje algebraico: racionalizo (conjugado)", sp.radsimp)):
            e2 = con_limite(lambda: fn(e), timeout=5)
            if e2 is None or e2 == e:
                continue
            if tipo_forma(e2, c, d) is None:
                val = _lim(e2, c, d)
                pasos.append((nombre, _lat_lim(e2, c, d) +
                              (r" = " + L(val) if val is not None else "")))
                return val
        return resolver_cociente(n, dd, c, d, pasos)
    if forma == "0·∞":
        ceros = [q for q in e.args if _lim(q, c, d) == 0]
        resto = [q for q in e.args if q not in ceros]
        A, B = sp.Mul(*ceros), sp.Mul(*resto)
        if B.is_rational_function(x) and not A.is_rational_function(x):
            # se deja la función trascendente arriba: A / (1/B)  → 0/0
            pasos.append(("Transformo el producto en un cociente: A / (1/B)",
                          r"A\cdot B=\frac{A}{1/B},\quad A=" + L(A) + r",\; B=" + L(B)))
            return resolver_cociente(A, 1 / B, c, d, pasos)
        pasos.append(("Transformo el producto en un cociente: B / (1/A)",
                      r"A\cdot B=\frac{B}{1/A},\quad A=" + L(A) + r",\; B=" + L(B)))
        return resolver_cociente(B, 1 / A, c, d, pasos)
    if forma == "∞−∞":
        comb = con_limite(lambda: sp.together(e), timeout=5)
        if comb is not None and comb != e and tipo_forma(comb, c, d) != "∞−∞":
            pasos.append(("Saco denominador común", _lat_lim(comb, c, d)))
            return resolver_limite(comb, c, d, pasos, prof + 1)
        return _lim(e, c, d)
    # potencias: 1^∞, 0^0, ∞^0  →  logaritmo
    b, ex = e.as_base_exp()
    pasos.append(("Potencia-exponencial: tomo logaritmo, y = f^g ⇒ ln y = g·ln f",
                  r"\ln y = " + L(ex) + r"\cdot\ln\!\left(" + L(b) + r"\right)"))
    lny = ex * sp.log(b)
    pasos.append(("Calculo el límite de ln y", ""))
    L_ln = resolver_limite(lny, c, d, pasos, prof + 1)
    if L_ln is None:
        return None
    val = sp.exp(L_ln) if L_ln != -sp.oo else sp.Integer(0)
    pasos.append(("Deshago el logaritmo: lím y = e^{lím ln y}",
                  r"\lim y = e^{" + L(L_ln) + r"} = " + L(val)))
    return val


def calc_limite(f_t, c_t, dir_t):
    f = parse_math(f_t, (x,))
    c = parse_math(c_t, ())
    d = _DIRS[dir_t]
    pasos = []
    mio = None
    try:
        mio = con_limite(lambda: resolver_limite(f, c, d, pasos), timeout=40)
    except Exception:
        mio = None
    ref = _lim(f, c, d)
    if ref is None:
        ref = con_limite(lambda: sp.limit(f, x, c, d if c not in (sp.oo, -sp.oo) else "+"),
                         timeout=10)
    return {"f": f, "c": c, "d": d, "pasos": pasos, "mio": mio, "ref": ref}


def ui_limites():
    st.markdown("Resolvé $\\lim_{x\\to c} f(x)$ mostrando el camino: forma "
                "indeterminada, salvataje algebraico, L'Hôpital o logaritmo.")
    f_t = st.text_input("f(x) =", "(1+2/x)^(5*x)", key="m9_f")
    c1, c2 = st.columns(2)
    c_t = c1.text_input("x → c   (c puede ser oo, -oo, pi, 2…)", "oo", key="m9_c")
    dir_t = c2.selectbox("Dirección", list(_DIRS.keys()), key="m9_d")
    if st.button("Resolver límite", key="m9_btn", type="primary"):
        try:
            with st.spinner("Resolviendo…"):
                _estado("m9_res", calc_limite(f_t, c_t, dir_t))
        except ValueError as e:
            _estado("m9_res", {"error": str(e)})
    r = _estado("m9_res")
    if not r:
        return
    if "error" in r:
        _error(r["error"])
        return
    st.latex(_lat_lim(r["f"], r["c"], r["d"]))
    for i, (tit, lat) in enumerate(r["pasos"], 1):
        st.markdown(f"**{i}. {tit}**")
        if lat:
            st.latex(lat)
    if r["ref"] is None:
        st.warning("SymPy no pudo determinar este límite (o no existe).")
        return
    if r["mio"] is None and r["pasos"]:
        st.caption("Los pasos anteriores son parciales; el valor final proviene de "
                   "la resolución simbólica.")
    st.success("Resultado: " + str(r["ref"]) + "   →   lím = " + N6(sp.N(r["ref"]))
               if r["ref"].is_number and r["ref"].is_finite else
               "Resultado: " + str(r["ref"]))
    st.latex(_lat_lim(r["f"], r["c"], r["d"]) + r" = " + L(r["ref"]))
    if r["mio"] is not None and r["mio"] != r["ref"]:
        try:
            if abs(float(sp.N(r["mio"] - r["ref"]))) > 1e-9:
                st.info("Nota: el camino paso a paso dio un valor distinto al de "
                        "la verificación simbólica; se toma esta última.")
        except Exception:
            pass


# ----------------------------------------------------------------------------
# 6. MARCO TEÓRICO (síntesis del material de la cátedra) E INTERFAZ PRINCIPAL
# ----------------------------------------------------------------------------
TEORIA = {
1: r"""
**Primitiva e integral indefinida.** $F$ es *primitiva* de $f$ en un intervalo $I$ si $F'(x)=f(x)$ para todo $x\in I$. Dos primitivas de la misma función difieren en una constante, por eso $\int f(x)\,dx=F(x)+C$ representa una **familia de curvas**.

- *Hallar todas* las primitivas: se deja $+C$. *Hallar una* primitiva: se fija un valor (por ejemplo $C=0$).
- **Integrar y derivar son operaciones inversas:** $\frac{d}{dx}\!\int f(x)\,dx=f(x)$ y $\int F'(x)\,dx=F(x)+C$.
- **Reglas básicas:** $\int x^n\,dx=\frac{x^{n+1}}{n+1}+C$ $(n\ne-1)$; $\int\frac{dx}{x}=\ln|x|+C$; linealidad $\int(\alpha f\pm\beta g)=\alpha\!\int f\pm\beta\!\int g$.
- **Inmediatas:** $\int e^{ax+b}dx=\frac{e^{ax+b}}{a}+C$; $\int a^x dx=\frac{a^x}{\ln a}+C$; $\int\cos=\operatorname{sen}$; $\int\operatorname{sen}=-\cos$; $\int\sec^2=\operatorname{tg}$; $\int\frac{dx}{1+x^2}=\operatorname{arctg}x$; $\int\frac{dx}{\sqrt{1-x^2}}=\operatorname{arcsen}x$.
- **Identidades útiles:** $\operatorname{sen}^2\alpha=\frac{1-\cos2\alpha}{2}$, $\cos^2\alpha=\frac{1+\cos2\alpha}{2}$, $1+\operatorname{tg}^2=\sec^2$, $1+\operatorname{cotg}^2=\operatorname{cosec}^2$. Para $\operatorname{sen}^3$ o $\cos^3$ (potencia impar): separar una unidad y usar $\operatorname{sen}^2=1-\cos^2$.
- **Sustitución:** si $F$ es primitiva de $f$, entonces $F(g(x))$ es primitiva de $f(g(x))\,g'(x)$. Se elige $u=g(x)$ y se reemplaza **todo** el integrando (incluido $dx$).
- **Por partes:** $\int u\,dv=uv-\int v\,du$. Se elige $u$ siguiendo el orden ILATE (inversas, logarítmicas, algebraicas, trigonométricas, exponenciales).
- **Criterio de la cátedra:** *acomodo e integro*: primero se acomoda el integrando (completar cuadrados, fracciones simples, identidades) y recién después se integra.
""",
2: r"""
**Problema de valor inicial.** Dada $f'(x)$ y un punto $P(x_0,y_0)$ de la gráfica de $f$:

1. Se integra: $f(x)=F(x)+C$ (familia de curvas que difieren en una constante vertical).
2. Se impone la condición: $f(x_0)=y_0\;\Rightarrow\;C=y_0-F(x_0)$.
3. Se escribe la curva **particular**: $f(x)=F(x)+C$.

- Si se conoce $f''$ se integra **dos veces** y se necesitan **dos** condiciones (por ejemplo $f'(x_1)$ y $f(x_0)$): primero se halla $f'$, se fija su constante y luego se integra $f'$.
- **Cinemática:** $v=s'$, $a=v'$. Dada la velocidad, la posición es $s(t)=\int v\,dt$ con $s(0)$ como condición inicial.
- **Razones de cambio:** si $\frac{dr}{dt}=g(t)$, la ley temporal es $r(t)=\int g\,dt$ con el dato inicial (o en un instante dado) del problema.
- Siempre conviene **verificar** derivando que $f'$ coincide con el dato y que la curva pasa por $P$.
""",
3: r"""
**Teorema Fundamental del Cálculo (TFC).** Si $f$ es continua en $[a,b]$ y $F(x)=\int_a^x f(t)\,dt$, entonces $F$ es derivable y $F'(x)=f(x)$ para todo $x\in[a,b]$.

**Con regla de la cadena:** $\dfrac{d}{dx}\displaystyle\int_{a(x)}^{b(x)} f(t)\,dt=f\big(b(x)\big)\,b'(x)-f\big(a(x)\big)\,a'(x)$.

- **Acomodo antes de derivar:** la variable debe estar en el límite *superior*; si está en el inferior se invierte con $\int_a^b=-\int_b^a$. Con dos límites funcionales se parte por una constante $c$: $\int_{a(x)}^{b(x)}=-\int_c^{a(x)}+\int_c^{b(x)}$.
- **Límites constantes:** si ambos límites son constantes la integral es un **número** y su derivada es $0$.
- **Ya no se integra:** el TFC reemplaza al cálculo de la primitiva; basta evaluar el integrando en los límites y multiplicar por la derivada de cada límite.
- **Constantes y factores:** una constante sumada se anula al derivar; si hay un factor $u(x)$ delante de la integral se usa la regla del producto.
- **Truco de paridad:** con límites simétricos $[-a,a]$ e integrando impar la integral vale $0$ (útil para evaluar $G(2)$ en funciones definidas por integrales).
- Si el integrando depende de $x$ además de $t$, se agrega $\int\frac{\partial f}{\partial x}dt$ (regla de Leibniz).
""",
4: r"""
**Regla de Barrow.** Si $f$ es continua en $[a,b]$ y $G$ es **cualquier** primitiva de $f$: $\displaystyle\int_a^b f(x)\,dx=G(b)-G(a)=\big[G(x)\big]_a^b$.

- **Propiedades:** linealidad; $\int_a^a f=0$; $\int_a^b f=-\int_b^a f$; aditividad $\int_a^b f=\int_a^c f+\int_c^b f$.
- **Monotonía:** si $f\le g$ en $[a,b]$ entonces $\int_a^b f\le\int_a^b g$; además $\big|\int_a^b f\big|\le\int_a^b|f|$.
- **Cambio de variable:** o se mantiene la variable original (se vuelve a $x$ antes de evaluar) o se **cambian los límites** a los valores de $u=g(x)$.
- **Simetría:** si $f$ es **par**, $\int_{-a}^a f=2\int_0^a f$; si es **impar**, $\int_{-a}^a f=0$.
- **Integral neta vs. área:** $\int_a^b f$ resta las zonas donde $f<0$. El área geométrica es $\int_a^b|f|$: se parte el intervalo en las raíces de $f$.
- **Parámetros:** en ecuaciones como $\int_{a}^{b} k\ln(3t)\,dt=9$ se calcula la integral en función del parámetro y se despeja.
""",
5: r"""
**Área entre curvas.** Se grafica siempre primero. Hay dos formas de cortar la región:

- **Rectángulos paralelos al eje $y$ ($dx$):** curvas como $y=f(x)$.
  $A=\displaystyle\int_a^b\big[\text{(arriba)}-\text{(abajo)}\big]\,dx$.
- **Rectángulos paralelos al eje $x$ ($dy$):** curvas como $x=g(y)$ (se despeja $x$).
  $A=\displaystyle\int_c^d\big[\text{(derecha)}-\text{(izquierda)}\big]\,dy$.

- **Puntos angulosos:** si en el recinto cambia el «techo» o el «piso» (la curva superior o inferior es otra) la región **se parte** en dos o más integrales: $A=\int_a^b+\int_b^c$.
- **Elección del corte:** se prefiere el que **no obliga a partir** la región o el que permite integrar con menos trabajo (a veces $dy$ evita un cambio de techo).
- Los límites de integración son las **abscisas (u ordenadas) de las intersecciones**: se igualan las curvas y se resuelve; se verifican en ambas ecuaciones.
- Cada integrando es positivo (arriba − abajo). Con funciones que cambian de signo el área es $\int|f|$.
""",
6: r"""
**Longitud de arco.** Si $f$ tiene derivada continua en $[a,b]$:
$$L=\int_a^b\sqrt{1+\big[f'(x)\big]^2}\,dx.$$
Si la curva es $x=g(y)$ con $g'$ continua en $[c,d]$: $L=\int_c^d\sqrt{1+\big[g'(y)\big]^2}\,dy$.

- **Estrategia:** derivar, elevar al cuadrado, sumar $1$ y **simplificar**: en los ejercicios de parcial el radicando casi siempre es un **cuadrado perfecto** (o permite una sustitución directa).
- $\sqrt{h^2}=|h|$: hay que analizar el signo de $h$ en el intervalo antes de quitar el módulo.
- La longitud depende sólo de la curva, no de cómo se la recorra: $L>0$.
- Con expresiones del tipo $\big(x^{3/2}\big)'=\tfrac32x^{1/2}$ conviene factorizar para llegar a $u=1+\tfrac94x$ (sustitución lineal).
""",
7: r"""
**Sólidos de revolución — fórmula general.** Al girar una región $R$ alrededor de un eje, cada corte perpendicular al eje es un **disco** o una **arandela** de área $A=\pi\big(R_{ext}^2-R_{int}^2\big)$, donde cada radio es la **distancia de la curva al eje**.

- **Eje horizontal $y=c$** $\Rightarrow$ cortes verticales $\Rightarrow$ se integra en $dx$:
  $V=\pi\displaystyle\int_a^b\Big(\big[f_{ext}(x)-c\big]^2-\big[f_{int}(x)-c\big]^2\Big)dx$.
- **Eje vertical $x=c$** $\Rightarrow$ cortes horizontales $\Rightarrow$ se integra en $dy$ con las funciones **inversas** $x=g(y)$:
  $V=\pi\displaystyle\int_c^d\Big(\big[g_{ext}(y)-c\big]^2-\big[g_{int}(y)-c\big]^2\Big)dy$.
- **Radio exterior** es el borde más lejano al eje, **interior** el más cercano. El eje **no debe atravesar** la región; si coincide con un borde, ese radio es $0$ (disco).
- **Puntos angulosos:** si cambia el borde exterior o interior, el volumen se parte en varias integrales.
- **Área de superficie de revolución:** $S=2\pi\displaystyle\int r\,ds$, con $r$ la distancia al eje y $ds=\sqrt{1+[f'(x)]^2}\,dx$.
- Graficar la región y marcar el eje es **obligatorio** antes de plantear.
""",
8: r"""
**Valor promedio.** Si $f$ es continua en $[a,b]$: $f_{prom}=\dfrac{1}{b-a}\displaystyle\int_a^b f(x)\,dx$.

**Teorema del Valor Medio del Cálculo Integral.** Si $f$ es continua en $[a,b]$ existe $c\in[a,b]$ tal que
$$f(c)=\frac{1}{b-a}\int_a^b f(x)\,dx,\qquad\text{es decir}\qquad \int_a^b f=f(c)\,(b-a).$$

- **Interpretación geométrica:** el área bajo $f$ coincide con el área del **rectángulo** de base $(b-a)$ y altura $f_{prom}$.
- **Cómo hallar $c$:** se resuelve $f(x)=f_{prom}$ y se conservan las soluciones que pertenecen a $[a,b]$ (puede haber más de una).
- **Problemas con parámetro:** si el dato es el valor promedio (por ejemplo «es igual a $\frac{\pi}{8b+40}$») se plantea la igualdad y se despeja el extremo o el parámetro.
- La demostración usa el Teorema de Weierstrass (máximo y mínimo) y el de los valores intermedios para una función continua.
""",
9: r"""
**Indeterminaciones y L'Hôpital.**

- **Formas:** $\frac00,\ \frac\infty\infty,\ 0\cdot\infty,\ \infty-\infty$ y las potencias $1^\infty,\ 0^0,\ \infty^0$.
- **Salvataje algebraico primero:** factorizar y simplificar, racionalizar con el conjugado, denominador común, dividir por la mayor potencia.
- **Regla de L'Hôpital:** si $\frac{f}{g}$ es $\frac00$ o $\frac\infty\infty$ y existe $\lim\frac{f'}{g'}$ (finito o infinito), entonces $\lim\frac fg=\lim\frac{f'}{g'}$. Se **deriva numerador y denominador por separado** (no es la derivada del cociente) y puede repetirse.
- **Producto $0\cdot\infty$:** se transforma en cociente, $f\cdot g=\dfrac{g}{1/f}$, para llegar a $\frac00$ o $\frac\infty\infty$.
- **Protocolo para potencias** $y=f^g$ ($1^\infty,0^0,\infty^0$): se toma logaritmo $\ln y=g\cdot\ln f$, se calcula $L=\lim\ln y$ (queda $0\cdot\infty$ → cociente → L'Hôpital) y se concluye $\lim y=e^{L}$.
- **Límite notable:** $\lim_{x\to\infty}\left(1+\frac kx\right)^{mx}=e^{km}$.
- **Con el TFC:** $\lim\frac{\int_0^{u(x)}f}{g(x)}$ de la forma $\frac00$ se resuelve con L'Hôpital derivando la integral con el TFC.
- **Convergencia:** una exponencial crece más rápido que cualquier potencia y una potencia más rápido que $\ln$: $\lim\frac{\ln x}{x^p}=0$, $\lim\frac{x^p}{e^x}=0$.
""",
}

AYUDA = """
**Cómo escribir las funciones** (variables: `x`, `y`, `t`):

| Quiero escribir | Se escribe |
|---|---|
| potencias | `x^2`, `x^(1/3)`, `e^(2*x)` |
| raíces | `sqrt(x)`, `cbrt(x)` |
| trigonométricas | `sen(x)`, `cos(x)`, `tg(x)`, `sec(x)`, `cotg(x)` |
| inversas | `arcsen(x)`, `arccos(x)`, `arctg(x)` |
| hiperbólicas | `senh(x)`, `cosh(x)`, `tgh(x)` |
| logaritmos | `ln(x)`, `log_7(x)`, `log2(x)` |
| valor absoluto | `abs(x)` |
| constantes | `pi`, `e`, `oo` (infinito) |

Usá `*` entre factores (`2*x`) y paréntesis en los argumentos (`sen(2*x)`).
Los puntos de la forma $P(x_0,y_0)$ y los límites aceptan expresiones: `pi/2`, `3/ln(2)^2`.
"""


def teoria(k):
    with st.expander("📘 Marco Teórico", expanded=False):
        st.markdown(TEORIA[k])


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📐", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(
        """<style>
        .block-container {padding-top: 1rem; padding-left: 0.7rem; padding-right: 0.7rem;}
        div[data-baseweb="tab-list"] {overflow-x: auto; flex-wrap: nowrap;}
        button[data-baseweb="tab"] {white-space: nowrap; padding: 0.4rem 0.6rem;}
        h1 {font-size: 1.6rem !important;}
        </style>""", unsafe_allow_html=True)
    st.title("📐 " + APP_TITLE)
    st.caption("Herramienta de estudio y verificación: calculá, graficá y comprobá "
               "tus resultados paso a paso.")
    with st.expander("ℹ️ Cómo ingresar las funciones", expanded=False):
        st.markdown(AYUDA)

    nombres = ["1 · Primitivas", "2 · PVI", "3 · TFC", "4 · Definidas",
               "5 · Área entre curvas", "6 · Long. de arco",
               "7 · Revolución 2D/3D", "8 · Valor promedio", "9 · Límites"]
    tabs = st.tabs(nombres)
    modulos = [ui_primitivas, ui_pvi, ui_tfc, ui_definidas, ui_area,
               ui_longitud, ui_volumen, ui_promedio, ui_limites]
    titulos = ["Primitivas e Integrales Indefinidas", "Problemas de Valor Inicial",
               "Derivadas de Funciones Integrales (TFC)", "Integrales Definidas",
               "Área entre Múltiples Curvas", "Longitud de Arco",
               "Volúmenes y Áreas de Superficies de Revolución",
               "Valor Promedio de una Función", "Límites e Indeterminaciones"]
    for k, (tab, fn, tit) in enumerate(zip(tabs, modulos, titulos), 1):
        with tab:
            st.subheader(tit)
            teoria(k)
            try:
                fn()
            except Exception as exc:  # protección general de la interfaz
                st.error("Ocurrió un error inesperado en este módulo. Revisá los "
                         f"datos ingresados. ({type(exc).__name__}: {exc})")


main()
