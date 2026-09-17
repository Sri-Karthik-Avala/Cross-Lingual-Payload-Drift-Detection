# made by - Karthik
import os
import sys
import json
import re
import html
import math
import time
import random
import zlib
import collections
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

T_START = time.time()
TRAIN_DEADLINE = 4400.0
SEED = 1234
ANC_MAX = 40
ECHO_MAX = 40
SPAN_W = 16
D_MODEL = 256
N_HEAD = 8
N_LAYER = 4
FFN = 1024
DROPOUT = 0.10
NGRAM_BUCKETS = 6000
MIN_FREQ = 2
BATCH = 32
LR = 2.5e-3
WD = 0.01
MAX_EPOCHS = 30
MIN_EPOCHS = 1
HOLDOUT = 1100
N_CYCLES = 3
CYCLE_FRAC = [0.52, 0.24, 0.24]
CYCLE_LR = [1.0, 0.45, 0.30]
FBETA = 1.0
NULL_BIAS_GRID = [-2.0, -1.5, -1.0, -0.5, 0.0]

torch.set_num_threads(max(1, min(10, os.cpu_count() or 10)))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")

PATCH_RE = re.compile(
    r"<PATCH>\s*(E\d{2})\s+(L\d{3})\s*<OLD>(.*?)</OLD>\s*<NEW>(.*?)</NEW>\s*</PATCH>", re.S)
ECHO_IDS = ["E01", "E02", "E03", "E04"]
MARK_TOK = ["<anc>", "<e1>", "<e2>", "<e3>", "<e4>"]


def canon(s):
    s = html.unescape(s if s is not None else "")
    s = unicodedata.normalize("NFKC", s)
    return " ".join(s.split())


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_gold(t):
    return [(m.group(1), m.group(2), canon(m.group(3)), canon(m.group(4)))
            for m in PATCH_RE.finditer(t)]


def find_sub(hay, ned, start=0):
    k = len(ned)
    if k == 0:
        return -1
    for i in range(start, len(hay) - k + 1):
        if hay[i:i + k] == ned:
            return i
    return -1


def ngram_ids(w):
    s = "<" + w + ">"
    out = set()
    for n in (3, 4):
        for i in range(len(s) - n + 1):
            out.add(zlib.crc32(s[i:i + n].encode("utf-8")) % NGRAM_BUCKETS)
    if not out:
        out.add(0)
    return sorted(out)


class Vocab:
    def __init__(self, rows):
        c = collections.Counter()
        for r in rows:
            for w in canon(r["anchor"]).split():
                c[w.lower()] += 1
            for e in r["echoes"]:
                for w in canon(e["text"]).split():
                    c[w.lower()] += 1
        self.itos = ["<pad>", "<unk>"] + list(MARK_TOK)
        self.itos += [w for w, n in c.most_common() if n >= MIN_FREQ]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.cache = {}

    def wid(self, w):
        return self.stoi.get(w, 1)

    def ng(self, w):
        v = self.cache.get(w)
        if v is None:
            v = ngram_ids(w)
            self.cache[w] = v
        return v


def build_sample(r, vocab, lane2id, with_gold):
    anc = canon(r["anchor"]).split()[:ANC_MAX]
    ech = {}
    for e in r["echoes"]:
        ech[e["echo_id"]] = canon(e["text"]).split()[:ECHO_MAX]
    views = [anc] + [ech.get(k, []) for k in ECHO_IDS]
    lower = [[w.lower() for w in v] for v in views]
    sets = [set(v) for v in lower]

    wids, ngl, viewv, posv, dfv, ancv = [], [], [], [], [], []
    marks, starts = [], []
    for vi, v in enumerate(views):
        marks.append(len(wids))
        wids.append(vocab.stoi[MARK_TOK[vi]])
        ngl.append([0])
        viewv.append(vi)
        posv.append(0)
        dfv.append(0)
        ancv.append(0)
        starts.append(len(wids))
        for pi, w in enumerate(v):
            lw = lower[vi][pi]
            wids.append(vocab.wid(lw))
            ngl.append(vocab.ng(lw))
            viewv.append(vi)
            posv.append(min(pi + 1, 63))
            dfv.append(sum(1 for u in range(5) if u != vi and lw in sets[u]))
            ancv.append(2 if vi == 0 else (1 if lw in sets[0] else 0))

    s = dict(wids=wids, ngl=ngl, view=viewv, pos=posv, df=dfv, anc=ancv,
             marks=marks, starts=starts, elen=[len(views[i + 1]) for i in range(4)],
             views=views, sid=r["sample_id"])

    if with_gold:
        gmap = {g[0]: g for g in parse_gold(r["repair_target"])}
        old_c, lane_t, copy_gold = [], [], []
        for ei, eid in enumerate(ECHO_IDS):
            g = gmap.get(eid)
            i, ot = -1, []
            if g is not None:
                ot = [w.lower() for w in g[2].split()]
                if 0 < len(ot) <= SPAN_W:
                    i = find_sub(lower[ei + 1], ot)
            if i < 0:
                old_c.append(0)
                lane_t.append(-100)
                copy_gold.append([])
                continue
            old_c.append(1 + i * SPAN_W + (len(ot) - 1))
            lane_t.append(lane2id[g[1]])
            nt = [w.lower() for w in g[3].split()]
            cg = []
            if 0 < len(nt) <= SPAN_W:
                for vj in range(5):
                    if vj == ei + 1:
                        continue
                    lv = lower[vj]
                    j = 0
                    while True:
                        j = find_sub(lv, nt, j)
                        if j < 0:
                            break
                        cg.append((starts[vj] + j, len(nt)))
                        j += 1
            copy_gold.append(cg[:8])
        s.update(old_c=old_c, lane_t=lane_t, copy_gold=copy_gold)
    return s


CAND_O = np.zeros(1 + ECHO_MAX * SPAN_W, dtype=np.int64)
CAND_W = np.zeros(1 + ECHO_MAX * SPAN_W, dtype=np.int64)
for _o in range(ECHO_MAX):
    for _w in range(1, SPAN_W + 1):
        _c = 1 + _o * SPAN_W + (_w - 1)
        CAND_O[_c] = _o
        CAND_W[_c] = _w
TCAND_O = torch.as_tensor(CAND_O)
TCAND_W = torch.as_tensor(CAND_W)


def collate(batch):
    B = len(batch)
    L = max(len(s["wids"]) for s in batch)
    wid = torch.zeros(B, L, dtype=torch.long)
    view = torch.zeros(B, L, dtype=torch.long)
    pos = torch.zeros(B, L, dtype=torch.long)
    df = torch.zeros(B, L, dtype=torch.long)
    anc = torch.zeros(B, L, dtype=torch.long)
    pad = torch.ones(B, L, dtype=torch.bool)
    ismark = torch.zeros(B, L, dtype=torch.bool)
    flat, offs = [], []
    for b, s in enumerate(batch):
        n = len(s["wids"])
        wid[b, :n] = torch.tensor(s["wids"])
        view[b, :n] = torch.tensor(s["view"])
        pos[b, :n] = torch.tensor(s["pos"])
        df[b, :n] = torch.tensor(s["df"])
        anc[b, :n] = torch.tensor(s["anc"])
        pad[b, :n] = False
        for m in s["marks"]:
            ismark[b, m] = True
        ngl = s["ngl"]
        for i in range(L):
            offs.append(len(flat))
            if i < n:
                flat.extend(ngl[i])
            else:
                flat.append(0)
    out = dict(
        wid=wid, view=view, pos=pos, df=df, anc=anc, pad=pad, ismark=ismark,
        ng_flat=torch.tensor(flat, dtype=torch.long),
        ng_off=torch.tensor(offs, dtype=torch.long),
        marks=torch.tensor([s["marks"][1:] for s in batch], dtype=torch.long),
        amark=torch.tensor([s["marks"][0] for s in batch], dtype=torch.long),
        starts=torch.tensor([s["starts"][1:] for s in batch], dtype=torch.long),
        elen=torch.tensor([s["elen"] for s in batch], dtype=torch.long),
        L=L, B=B)
    if "old_c" in batch[0]:
        out["old_c"] = torch.tensor([s["old_c"] for s in batch], dtype=torch.long)
        out["lane_t"] = torch.tensor([s["lane_t"] for s in batch], dtype=torch.long)
        cg = torch.zeros(B, 4, L * SPAN_W, dtype=torch.bool)
        for b, s in enumerate(batch):
            for e in range(4):
                for (p, w) in s["copy_gold"][e]:
                    cg[b, e, p * SPAN_W + (w - 1)] = True
        out["copy_gold"] = cg
    return out


class Model(nn.Module):
    def __init__(self, nvocab, nlane):
        super().__init__()
        d = D_MODEL
        self.nlane = nlane
        self.wemb = nn.Embedding(nvocab, d, padding_idx=0)
        self.nemb = nn.EmbeddingBag(NGRAM_BUCKETS, d, mode="mean")
        self.vemb = nn.Embedding(5, d)
        self.pemb = nn.Embedding(64, d)
        self.demb = nn.Embedding(5, d)
        self.aemb = nn.Embedding(3, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(DROPOUT)
        layer = nn.TransformerEncoderLayer(d, N_HEAD, FFN, dropout=DROPOUT,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, N_LAYER)
        self.encnorm = nn.LayerNorm(d)
        self.os = nn.Linear(d, 1)
        self.oe = nn.Linear(d, 1)
        self.olen = nn.Parameter(torch.zeros(SPAN_W + 1))
        self.lane = nn.Sequential(nn.Linear(5 * d, d), nn.GELU(),
                                  nn.Dropout(DROPOUT), nn.Linear(d, nlane))
        self.lemb = nn.Embedding(nlane, d)
        self.q = nn.Sequential(nn.Linear(5 * d, d), nn.GELU(), nn.Linear(d, d))
        self.cs = nn.Linear(d, d)
        self.ce = nn.Linear(d, d)
        self.clen = nn.Parameter(torch.zeros(SPAN_W + 1))

    def encode(self, bt):
        B, L = bt["B"], bt["L"]
        ng = self.nemb(bt["ng_flat"], bt["ng_off"]).view(B, L, -1)
        x = (self.wemb(bt["wid"]) + ng + self.vemb(bt["view"]) + self.pemb(bt["pos"])
             + self.demb(bt["df"]) + self.aemb(bt["anc"]))
        x = self.drop(self.norm(x))
        h = self.enc(x, src_key_padding_mask=bt["pad"])
        return self.encnorm(h)

    def old_scores(self, h, bt):
        B, L = bt["B"], bt["L"]
        s = self.os(h).squeeze(-1)
        t = self.oe(h).squeeze(-1)
        co = TCAND_O.view(1, 1, -1)
        cw = TCAND_W.view(1, 1, -1)
        isnull = cw == 0
        st = (bt["starts"].unsqueeze(-1) + co).clamp(0, L - 1)
        en = (bt["starts"].unsqueeze(-1) + co + cw - 1).clamp(0, L - 1)
        mk = bt["marks"].unsqueeze(-1)
        st = torch.where(isnull, mk, st)
        en = torch.where(isnull, mk, en)
        sc = (s.unsqueeze(1).expand(B, 4, L).gather(2, st)
              + t.unsqueeze(1).expand(B, 4, L).gather(2, en)
              + self.olen[TCAND_W].view(1, 1, -1))
        valid = isnull | ((co + cw) <= bt["elen"].unsqueeze(-1))
        return sc.masked_fill(~valid, -1e9)

    def _feat(self, h, bt, si, ei):
        B = bt["B"]
        idx = torch.arange(B).unsqueeze(1).expand(B, 4)
        hi = h[idx, si]
        hj = h[idx, ei]
        hm = h[idx, bt["marks"]]
        ha = h[torch.arange(B), bt["amark"]].unsqueeze(1).expand(B, 4, h.shape[-1])
        return hi, hj, hm, ha

    def lane_logits(self, h, bt, si, ei):
        hi, hj, hm, ha = self._feat(h, bt, si, ei)
        return self.lane(torch.cat([hi, hj, hm, ha, hi * hj], -1))

    def copy_scores(self, h, bt, si, ei, lane_logits):
        B, L = bt["B"], bt["L"]
        hi, hj, hm, ha = self._feat(h, bt, si, ei)
        lc = torch.softmax(lane_logits, -1) @ self.lemb.weight
        q = self.q(torch.cat([hi, hj, hm, ha, lc], -1))
        a = torch.einsum("bed,bld->bel", q, self.cs(h)) / math.sqrt(D_MODEL)
        b = torch.einsum("bed,bld->bel", q, self.ce(h)) / math.sqrt(D_MODEL)
        w = torch.arange(1, SPAN_W + 1)
        endi = torch.arange(L).view(L, 1) + w.view(1, -1) - 1
        ok = endi < L
        endc = endi.clamp(max=L - 1).reshape(1, 1, -1).expand(B, 4, L * SPAN_W)
        sc = (a.unsqueeze(-1) + b.gather(2, endc).view(B, 4, L, SPAN_W)
              + self.clen[w].view(1, 1, 1, -1))
        allowed = (~bt["pad"]) & (~bt["ismark"])
        allowed = allowed.unsqueeze(1) & (bt["view"].unsqueeze(1)
                                          != (torch.arange(4) + 1).view(1, 4, 1))
        cum = torch.cumsum(allowed.long(), -1)
        cum = torch.cat([torch.zeros(B, 4, 1, dtype=cum.dtype), cum], -1)
        tot = cum.gather(2, (endc + 1)).view(B, 4, L, SPAN_W) - cum[:, :, :L].unsqueeze(-1)
        valid = (tot == w.view(1, 1, 1, -1)) & ok.view(1, 1, L, SPAN_W)
        return sc.masked_fill(~valid, -1e9).view(B, 4, L * SPAN_W)


def gold_span_idx(bt, gc):
    si = torch.where(gc == 0, bt["marks"], bt["starts"] + TCAND_O[gc])
    ei = torch.where(gc == 0, bt["marks"], bt["starts"] + TCAND_O[gc] + TCAND_W[gc] - 1)
    return si, ei


def compute_loss(model, bt):
    h = model.encode(bt)
    B = bt["B"]
    osc = model.old_scores(h, bt)
    l_old = F.cross_entropy(osc.reshape(B * 4, -1), bt["old_c"].reshape(-1))
    si, ei = gold_span_idx(bt, bt["old_c"])
    ll = model.lane_logits(h, bt, si, ei)
    lt = bt["lane_t"].reshape(-1)
    l_lane = (F.cross_entropy(ll.reshape(B * 4, -1), lt, ignore_index=-100)
              if bool((lt >= 0).any()) else torch.zeros(()))
    csc = model.copy_scores(h, bt, si, ei, ll.detach())
    cg = bt["copy_gold"]
    hasg = cg.any(-1)
    if bool(hasg.any()):
        l_copy = (torch.logsumexp(csc, -1)
                  - torch.logsumexp(csc.masked_fill(~cg, -1e9), -1))[hasg].mean()
    else:
        l_copy = torch.zeros(())
    return l_old + l_lane + l_copy


@torch.no_grad()
def decode(model, snaps, samples, lanes, null_bias, min_patches=0, bs=64):
    model.eval()
    order = sorted(range(len(samples)), key=lambda i: len(samples[i]["wids"]))
    out = [""] * len(samples)
    params = list(model.parameters())
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        bt = collate([samples[i] for i in idx])
        hs = []
        osc = None
        for sd in snaps:
            for p_, v_ in zip(params, sd):
                p_.copy_(v_)
            h = model.encode(bt)
            hs.append(h)
            o = model.old_scores(h, bt)
            osc = o if osc is None else osc + o
        osc = osc / len(snaps)
        osc[:, :, 0] = osc[:, :, 0] + null_bias
        pc = osc.argmax(-1)
        if min_patches > 0:
            top = osc[:, :, 1:].max(-1)
            order_e = (top.values - osc[:, :, 0]).argsort(-1, descending=True)
            for b in range(pc.shape[0]):
                cnt = int((pc[b] > 0).sum())
                for r in range(4):
                    if cnt >= min_patches:
                        break
                    e = int(order_e[b, r])
                    if pc[b, e] == 0 and float(top.values[b, e]) > -1e8:
                        pc[b, e] = top.indices[b, e] + 1
                        cnt += 1
        si, ei = gold_span_idx(bt, pc)
        ll = csc = None
        for sd, h in zip(snaps, hs):
            for p_, v_ in zip(params, sd):
                p_.copy_(v_)
            a = model.lane_logits(h, bt, si, ei)
            b = model.copy_scores(h, bt, si, ei, a)
            ll = a if ll is None else ll + a
            csc = b if csc is None else csc + b
        lp = ll.argmax(-1).tolist()
        cp = csc.argmax(-1).tolist()
        pcl = pc.tolist()
        for bi, i in enumerate(idx):
            s = samples[i]
            flat = []
            for v in s["views"]:
                flat.append(None)
                flat.extend(v)
            parts = []
            for e in range(4):
                c = pcl[bi][e]
                if c == 0:
                    continue
                o = int(CAND_O[c])
                w = int(CAND_W[c])
                ev = s["views"][e + 1]
                if o + w > len(ev):
                    continue
                kk = cp[bi][e]
                p, ww = kk // SPAN_W, kk % SPAN_W + 1
                if p + ww > len(flat) or any(x is None for x in flat[p:p + ww]):
                    continue
                parts.append("<PATCH> %s %s <OLD> %s </OLD> <NEW> %s </NEW> </PATCH>"
                             % (ECHO_IDS[e], lanes[lp[bi][e]],
                                " ".join(ev[o:o + w]), " ".join(flat[p:p + ww])))
            out[i] = " ".join(parts)
    return out


def cycle_bounds(n_epochs):
    if n_epochs < 2 * N_CYCLES:
        return [n_epochs]
    out, acc = [], 0
    for i in range(N_CYCLES):
        acc += max(1, int(round(CYCLE_FRAC[i] * n_epochs)))
        out.append(min(n_epochs, acc))
    out[-1] = n_epochs
    return sorted(set(out))


def lr_factor(step, warm, per_epoch, bounds):
    if step < warm:
        return (step + 1) / warm
    ep = step / per_epoch
    lo = 0.0
    for i, b in enumerate(bounds):
        if ep < b:
            scale = CYCLE_LR[min(i, len(CYCLE_LR) - 1)]
            pr = min(1.0, max(0.0, (ep - lo) / max(1e-6, b - lo)))
            return scale * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * pr)))
        lo = b
    return CYCLE_LR[-1] * 0.01


def key_fbeta(preds, golds, beta):
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        sp = {(x[0], x[1]) for x in parse_gold(p)}
        sg = {(x[0], x[1]) for x in parse_gold(g)}
        tp += len(sp & sg)
        fp += len(sp - sg)
        fn += len(sg - sp)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def main():
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])

    train_rows = read_jsonl(public_dir / "train.jsonl")
    test_rows = read_jsonl(public_dir / "test.jsonl")
    submission = pd.read_csv(public_dir / "sample_submission.csv")
    submission["prediction"] = ""
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_out, index=False)

    lanes = sorted({m.group(2) for r in train_rows
                    for m in PATCH_RE.finditer(r["repair_target"])})
    lane2id = {l: i for i, l in enumerate(lanes)}
    min_patches = min(len(parse_gold(r["repair_target"])) for r in train_rows)

    const = collections.defaultdict(list)
    for r in train_rows:
        const[tuple(sorted((g[0], g[1]) for g in parse_gold(r["repair_target"])))].append(r)
    keys = sorted(const.keys())
    rng = random.Random(SEED)
    rng.shuffle(keys)
    hold, fit = [], []
    for k in keys:
        if len(hold) < HOLDOUT:
            hold.extend(const[k])
        else:
            fit.extend(const[k])

    vocab = Vocab(fit)
    fit_s = [build_sample(r, vocab, lane2id, True) for r in fit]
    hold_s = [build_sample(r, vocab, lane2id, True) for r in hold]
    test_s = [build_sample(r, vocab, lane2id, False) for r in test_rows]
    hold_gold = [r["repair_target"] for r in hold]

    model = Model(len(vocab.itos), len(lanes))
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)

    nb_per_epoch = max(1, (len(fit_s) + BATCH - 1) // BATCH)
    planned = MAX_EPOCHS
    bounds = cycle_bounds(planned)
    warm = max(1, int(0.8 * nb_per_epoch))
    order = list(range(len(fit_s)))
    snaps = []
    ep_hist = []
    step = 0
    done = 0
    ep_time = None

    while done < planned:
        if done >= MIN_EPOCHS:
            if ep_time is not None and time.time() - T_START + ep_time > TRAIN_DEADLINE:
                break
        model.train()
        t1 = time.time()
        rng.shuffle(order)
        batches = []
        for i in range(0, len(order), BATCH * 24):
            ch = sorted(order[i:i + BATCH * 24], key=lambda j: len(fit_s[j]["wids"]))
            batches.extend([ch[j:j + BATCH] for j in range(0, len(ch), BATCH)])
        rng.shuffle(batches)
        for bidx in batches:
            bt = collate([fit_s[j] for j in bidx])
            loss = compute_loss(model, bt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for g in opt.param_groups:
                g["lr"] = LR * lr_factor(step, warm, nb_per_epoch, bounds)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        done += 1
        ep_time = time.time() - t1
        ep_hist.append(ep_time)
        est = max(ep_hist[-2:])
        room = int((TRAIN_DEADLINE - (time.time() - T_START)) / est)
        planned = max(done, min(MAX_EPOCHS, done + max(0, room)))
        bounds = cycle_bounds(planned)
        if done in bounds or done == planned:
            snaps.append([p.detach().clone() for p in params])
        print("epoch %d/%d  %.0fs  elapsed %.0fs  snaps %d"
              % (done, planned, ep_time, time.time() - T_START, len(snaps)), flush=True)

    if not snaps:
        snaps.append([p.detach().clone() for p in params])
    snaps = snaps[-1:]
    best_nb, best_v = NULL_BIAS_GRID[0], -1.0
    for nb in NULL_BIAS_GRID:
        v = key_fbeta(decode(model, snaps, hold_s, lanes, nb, min_patches), hold_gold, FBETA)
        print("null_bias %+.2f  fbeta %.4f" % (nb, v), flush=True)
        if v > best_v:
            best_v, best_nb = v, nb

    preds = decode(model, snaps, test_s, lanes, best_nb, min_patches)
    pmap = {s["sid"]: p for s, p in zip(test_s, preds)}
    submission["prediction"] = [pmap.get(sid, "") for sid in submission["sample_id"]]
    submission.to_csv(submission_out, index=False)
    print("done %.0fs  null_bias %+.2f" % (time.time() - T_START, best_nb), flush=True)


main()
