# tensorflow-cljs

ClojureScript (Node.js) wrapper of **TensorFlow.js** (`@tensorflow/tfjs`
4.22.0, pure-JS CPU backend): the tensor op algebra, layers and models
(sequential + functional), optimizers / losses / metrics, `tf.data`
pipelines, model save/load and training callbacks — the whole main tfjs
surface, in the house style of the sibling `*-cljs` libraries.

The authoritative spec is `docs/tensorflow-api-inventory.org` (phase 1
research: package census, op / layer / model / data / io catalogs, error
taxonomy, pitfalls, numbered uncertainties). The only runtime npm dependency
is `@tensorflow/tfjs` itself.

## Status

| Phase | Content | State |
|-------|---------|-------|
| 0 | scaffold: shadow-cljs `:node-test` / `:dev` builds, `run-tests.sh`, `util` (kebab/camel, dtypes, errors) | done |
| 1 | API inventory (`docs/tensorflow-api-inventory.org`) | done |
| 2 | `binding` (the ONE interop namespace) + `core`: tensor creation, data access, memory (tidy/dispose), the op algebra, backend control | planned |
| 3 | `layers` + `models`: layer catalog from kebab config maps, sequential / functional models, compile / fit / predict / evaluate, callbacks, `train` (optimizers, losses, metrics, custom loops) | planned |
| 4 | `data` (tf.data over core.async / seqables) + `io` (save / load incl. a node file-system handler) | planned |
| 5 | live smoke (`dev/`): end-to-end train + save + reload on the CPU backend, perf baseline | planned |

`./run-tests.sh` — 8 tests / 88 assertions (phase 0), no network.

## Hard style rules (non-negotiable)

- **ClojureScript only.** No `^js` type hints anywhere. No JavaScript interop at
  call sites: every `js/`, `.-prop`, `.method`, `#js`, `clj->js` that touches
  tfjs lives in the single binding namespace (`tensorflow-cljs.binding`,
  phase 2). Everything else is plain CLJS over that namespace.
- **Kebab-case keyword maps in and out.** `{:units 4 :activation :relu
  :kernel-initializer :glorot-uniform}` → tfjs `{units: 4, activation: "relu",
  kernelInitializer: "glorotUniform"}`; histories / logs / memory stats come
  back as `{:val-loss [...] :num-tensors 3}`. Enum keywords become the tfjs
  registry strings (`util/enum-value`).
- **promesa promises** for every asynchronous model operation (`fit!`,
  `load-model!`, `data`, `evaluate-dataset!`, `save!`); synchronous tfjs calls
  (`predict`, `data-sync`, op algebra) stay synchronous.
- **Tensors are opaque values.** The library never converts a tensor to CLJS
  data unless asked (`->clj`, `->vec`); it never exposes tfjs classes.
- **Zero runtime npm deps beyond `@tensorflow/tfjs`** (promesa and core.async
  are Clojars/ClojureScript deps).

## Backend note

Pure `@tensorflow/tfjs` on Node.js registers **only the `cpu` backend**
(verified on node v22.23.2: `webgl` never registers — no WebGL context —
and `setBackend "webgl"` rejects with `Backend name 'webgl' not found in
registry`). It needs no native build and installs in ~10 s (272 MB under
`node_modules/@tensorflow/`, 141 MB of which are the umbrella bundles). CPU
performance is JavaScript-speed (512×512 float32 matmul ≈ 0.6 s; a 2-layer
MLP over 1024 samples × 3 epochs ≈ 0.7 s). `@tensorflow/tfjs-node` (native
libtensorflow kernels, N-API 8 prebuilt binary for linux-x64 exists for
4.22.0) is a later opt-in backend, see inventory §1.4 and §11.

## Quick start (target API, phase 2+)

```clojure
(require '[tensorflow-cljs.core :as tf]
         '[tensorflow-cljs.models :as models]
         '[tensorflow-cljs.layers :as layers]
         '[promesa.core :as p])

(def xs (tf/tensor [[0 0] [0 1] [1 0] [1 1]]))          ; opaque tensor, float32 [4 2]
(def ys (tf/tensor [[0] [1] [1] [0]]))

(def m (models/sequential
        [(layers/dense {:units 8 :activation :tanh :input-shape [2]})
         (layers/dense {:units 1 :activation :sigmoid})]))

(models/compile! m {:optimizer :adam :loss :binary-crossentropy :metrics [:accuracy]})

(p/let [history (models/fit! m xs ys {:epochs 50 :batch-size 4 :verbose 0
                                      :callbacks {:on-epoch-end (fn [epoch logs] nil)}})]
  (println (:loss (:history history)))
  (println (tf/->vec (models/predict m xs))))
```

## Errors

Rejections / throws are `ex-info` values (`tensorflow-cljs.util/error-data`):
`:tensorflow-cljs/tf-error` (a tfjs error, `:code` one of `:value-error`
`:runtime-error` `:not-implemented-error` `:attribute-error` `:index-error`
`:shape-error` `:assertion-error` `:unknown`, `:name` = the JS constructor
name, `:cause` = the original error), `:argument-error` (caught before any
tfjs call), `:io-error` (save / load), `:backend-error`.

## Development

```
./run-tests.sh                                  # exec's node at the end — never chain it mid-command
npx shadow-cljs watch dev && node target/dev.js # REPL runtime
```
