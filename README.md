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
| 2 | `core` (the ONE interop namespace): tensor creation, data access, memory (tidy/dispose), the whole op algebra (§2.2 census incl. image / linalg / losses / signal / sparse / string / fused), backend control, optimizers + gradients | done |
| 3 | `layers` (66 constructors + 10 aliases, registries as data) + `models` (sequential / functional, compile / fit / predict / evaluate, callbacks, early stopping, tf.data training, save / load with the library's node-fs handler, in-memory artifacts, topology JSON) + `data` (array / generator / func / csv / zip sources, combinators, promise consumers) | done |
| 4 | `io` extras (http handler options, graph models), core.async channel over datasets | planned |
| 5 | live smoke (`dev/`): end-to-end train + save + reload on the CPU backend, perf baseline | planned |
| 6 | full documentation | planned |

`./run-tests.sh` — 72 tests / 892 assertions (phases 0–3), no network, ≈2 s of
test time on the CPU backend (XOR trains to 100 % inside the suite).

## Hard style rules (non-negotiable)

- **ClojureScript only.** No `^js` type hints anywhere. No JavaScript interop at
  call sites: every `js/`, `js-invoke`, `goog.object`, `#js`, `clj->js` that
  touches tfjs (or node `fs`) lives in the single namespace
  `tensorflow-cljs.core`. `layers`, `models` and `data` are plain CLJS over
  `core` and only receive or return opaque values.
- **Kebab-case keyword maps in and out.** `{:units 4 :activation :relu
  :kernel-initializer :glorot-uniform}` → tfjs `{units: 4, activation: "relu",
  kernelInitializer: "glorotUniform"}`; histories / logs / memory stats come
  back as `{:val-loss [...] :num-tensors 3}`. Enum keywords become the tfjs
  registry strings (`util/enum-value`, with the one `:gelu-new → "gelu_new"`
  exception); losses / metrics / optimizers go through registries in
  `models` (`:mean-squared-error` or `:mse` → `"meanSquaredError"`).
- **promesa promises** for every asynchronous model operation (`fit!`,
  `fit-dataset!`, `evaluate-dataset!`, `train-on-batch!`, `save!`, `load!`,
  `data!`, `->vec!`, dataset consumers); `predict` / `evaluate` and the op
  algebra are synchronous, with `predict!` / `evaluate!` promise twins.
- **Tensors are opaque values.** The library never converts a tensor to CLJS
  data unless asked (`->vec` / `->clj`, `data-sync`); it never exposes tfjs
  classes. Layers, models, symbolic tensors, optimizers, datasets and
  artifacts are opaque the same way.
- **Zero runtime npm deps beyond `@tensorflow/tfjs`** (promesa is a Clojars
  dep).

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

## Quick start

```clojure
(require '[tensorflow-cljs.core :as tf]
         '[tensorflow-cljs.layers :as layers]
         '[tensorflow-cljs.models :as models]
         '[tensorflow-cljs.data :as data]
         '[promesa.core :as p])

;; tensors: numbers become float32, nesting gives the shape
(def a (tf/tensor [[1 2] [3 4]]))                  ; opaque, shape [2 2]
(tf/->vec (tf/mat-mul a (tf/tensor [[5 6] [7 8]]))) ; => [[19 22] [43 50]]
(tf/->vec (tf/add a 1))                            ; broadcasting => [[2 3] [4 5]]
(tf/->vec (tf/equal a 2))                          ; bool => [[false true] [false false]]
(tf/tidy (fn [] {:s (tf/sum (tf/square a))}))      ; only the returned tensors survive

;; models
(def xs (tf/tensor [[0 0] [0 1] [1 0] [1 1]]))
(def ys (tf/tensor [[0] [1] [1] [0]]))

(def m (models/sequential
        [(layers/dense {:units 8 :activation :tanh :input-shape [2]})
         (layers/dense {:units 1 :activation :sigmoid})]))

(models/compile-model! m {:optimizer {:type :adam :learning-rate 0.1}
                          :loss :binary-crossentropy
                          :metrics [:accuracy]})

(p/let [history (models/fit! m xs ys {:epochs 150 :batch-size 4 :yield-every :never
                                      :callbacks {:on-epoch-end (fn [epoch logs] nil)}})
        _       (models/save! m "/tmp/xor-model")   ; model.json + weights.bin
        m2      (models/load! "/tmp/xor-model")]
  (println (last (get-in history [:history :loss])))
  (println (models/evaluate m2 xs ys))              ; => {:loss 0.0.. :acc 1}
  (println (tf/->vec (models/predict m2 xs))))

;; tf.data
(def ds (-> (data/from-vec [{:xs [0 0] :ys [0]} {:xs [0 1] :ys [1]}])
            (data/shuffle 2 {:seed "x"})
            (data/batch 2)))
(models/fit-dataset! m ds {:epochs 2})
```

## Errors

Rejections / throws are `ex-info` values (`tensorflow-cljs.util/error-data`):
`:tensorflow-cljs/tf-error` (a tfjs error, `:code` one of `:value-error`
`:runtime-error` `:not-implemented-error` `:attribute-error` `:index-error`
`:shape-error` `:assertion-error` `:unknown`, `:name` = the JS constructor
name, `:cause` = the original error), `:argument-error` (caught before any
tfjs call: unknown dtype / registry keyword, ragged tensor data, a shape that
does not match the value count, a dataset of unknown size without
`:batches-per-epoch`), `:io-error` (save / load: missing model.json or weight
shard, fetch failures), `:backend-error` (`set-backend!`).

## Gotchas (see inventory §10)

- Every number is float32 (`0.1` reads back as `0.10000000149011612`); ask for
  `:int32` explicitly (wraps at 2³¹, truncates fractions).
- tfjs 4.22 does **not** validate ragged nested input; the wrapper does
  (`:argument-error`).
- `strides` follow tfjs (rank-1 entries, trailing 1 omitted).
- Only `tidy`, `dispose!` and `models/dispose!` free memory; `evaluate`
  reads and disposes its scalars, `predict` leaves its output to you.
- `save!` without `{:include-optimizer true}` does not store the training
  config: the loaded model must be compiled again.

## Development

```
./run-tests.sh                                  # exec's node at the end — never chain it mid-command
npx shadow-cljs watch dev && node target/dev.js # REPL runtime
```
