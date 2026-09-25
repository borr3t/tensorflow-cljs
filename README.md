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
| 4 | leftovers: `models/load-graph!` + `graph-*` (inference-only converter output through the same offline artifacts path), `data/->channel!` (core.async channel over any dataset, error as last item, close-to-cancel) + `channel->vec!`; fixes found by the smoke: `BaseCallback` reached through the prototype chain (the umbrella does not export it), `:on-yield` interval forwarded to tfjs's callback constructor, `models/dispose!` also frees the optimizer's kept scalars; inventory §11b records the corrections | done |
| 5 | live smoke (`dev/tensorflow_cljs/smoke.cljs`, `:smoke` build): real compute on the CPU backend + real files in a temp dir, 9 PASS/FAIL sections 0–8 with per-section tensor accounting, ≈1.6 s, no network | done |
| 6 | user manual (`docs/tensorflow-cljs.org`, 13 chapters + API index) + this README | done |

`./run-tests.sh` — 74 tests / 935 assertions, no network, ≈2 s of test time on the CPU
backend (XOR trains to 100 % inside the suite). `npx shadow-cljs compile smoke && node
target/smoke.js` — the live smoke, 9/9 sections.

Recorded as debt (manual, "Design decisions"): custom IO handler objects (the
in-memory `save-artifacts!` / `load-artifacts!` pair covers the use), http *save*,
tfjs-node as a tested backend, `GraphModel` intermediate-tensor extras, quantized /
multi-shard *writing*.

## Documentation

- [`docs/tensorflow-cljs.org`](docs/tensorflow-cljs.org) — the user manual: install and
  quick start, tensors (creation, the dtype table and its traps, opaque values,
  readers), the op algebra (conventions, how to read the census, examples, custom
  training loops), memory (`tidy` / `dispose!` / `memory`, what is yours to free), the
  layers catalog and config translation, models (compile registries, `fit!` options,
  callback maps and `:on-yield` semantics, early stopping, sync / promise twins, save /
  load with `:include-optimizer` and sharding, graph models), data pipelines and the
  core.async channel, backend notes, error handling, design decisions with the phase-4
  debt, testing and the live smoke, and a per-namespace API index.
- [`docs/tensorflow-api-inventory.org`](docs/tensorflow-api-inventory.org) — the research
  the library is built on: package census, op / layer / model / data / io catalogs, error
  taxonomy, pitfalls, the numbered uncertainties with the house position on each, and
  the phase-2/3/4 corrections (§11b) found live.

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
(tf/->vec (tf/tensor [1.9 -2.1] :int32))           ; => [1 -2]
(tf/tidy (fn [] {:s (tf/sum (tf/square a))}))      ; only the returned tensors survive

;; models (the seeded XOR of the suite and the smoke: 100 % in 150 epochs, ≈0.5 s)
(def xs (tf/tensor [[0 0] [0 1] [1 0] [1 1]]))
(def ys (tf/tensor [[0] [1] [1] [0]]))

(def m (models/sequential
        [(layers/dense {:units 8 :activation :tanh :input-shape [2]
                        :kernel-initializer {:type :glorot-uniform :seed 7}})
         (layers/dense {:units 1 :activation :sigmoid})]))

(models/compile-model! m {:optimizer {:type :adam :learning-rate 0.1}
                          :loss :binary-crossentropy
                          :metrics [:accuracy]})

(p/let [history (models/fit! m xs ys {:epochs 150 :batch-size 4 :yield-every :never
                                      :callbacks {:on-epoch-end (fn [epoch logs] nil)}})
        _       (models/save! m "models/xor")                          ; model.json + weights.bin
        _       (models/save! m "models/xor-opt" {:include-optimizer true})
        m2      (models/load! "models/xor")                            ; predicts identically, NOT compiled
        m3      (models/load! "models/xor-opt")]                       ; compiled: fit! continues training
  (println (last (get-in history [:history :loss])))                  ; ≈ 0.0007
  (println (models/evaluate m xs ys))                                 ; => {:loss 0.0007 :acc 1}
  (println (tf/->vec (models/predict m2 xs)))
  (models/dispose! m2))

;; tf.data: a CSV file -> batches -> training, or a core.async channel
(def ds (-> (data/csv "data/xor.csv" {:column-configs {"label" {:is-label true}}})
            (data/map (fn [{:keys [xs ys]}] {:xs [(:x1 xs) (:x2 xs)] :ys [(:label ys)]}))
            (data/shuffle 8 {:seed "x"})
            (data/batch 4)))
(models/fit-dataset! m (data/repeat ds) {:epochs 2 :batches-per-epoch 2})
(p/let [batches (data/channel->vec! (data/->channel! (data/take ds 2)))]
  (tf/shape (:xs (first batches))))                                    ; => [4 2]

;; a converter-exported graph model (inference only)
(p/let [g (models/load-graph! "models/mobilenet")]
  (models/graph-predict g (tf/zeros [1 224 224 3])))
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
  config: the loaded model must be compiled again. With it, tfjs leaks one int32
  scalar per save (inventory §11b.4).
- `:on-yield` fires only for a numeric / `:auto` `:yield-every` (debounced by wall
  time); `:epoch` / `:batch` never call it and `:never` + `:on-yield` is rejected by
  tfjs (§11b.2).
- `fit-dataset!` reuses one iterator across epochs: use `(data/repeat ds)` with
  `:batches-per-epoch` for more than one epoch (§11b.5).
- `models/dispose!` also frees the optimizer (SGD / Momentum keep plain scalars that
  `dispose-variables!` does not see, §11b.3); pass `{:keep-optimizer true}` for a
  shared optimizer object.

## Development

```
./run-tests.sh                                          # exec's node at the end — never chain it mid-command
npx shadow-cljs compile smoke && node target/smoke.js   # live smoke (TFCLJS_SMOKE_STACK=1 for JS stacks on failures)
npx shadow-cljs watch dev && node target/dev.js         # REPL runtime
```
