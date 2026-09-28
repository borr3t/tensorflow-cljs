# tensorflow-cljs

An idiomatic ClojureScript (Node.js) wrapper of
[TensorFlow.js](https://www.tensorflow.org/js) (`@tensorflow/tfjs` 4.22): tensors and the
whole op algebra, layers and models (sequential and functional), optimizers, losses and
metrics, `tf.data` pipelines, saving and loading, training callbacks. Configs and results
are maps of kebab-cased keywords, asynchronous operations return
[promesa](https://github.com/funcool/promesa) promises, tensors, layers, models and
datasets are opaque values, and everything is computed in the process that calls it: there
is no service behind the library and no account to open.

## Requirements

Node.js 22+, Java 11+ for shadow-cljs, and `@tensorflow/tfjs` in the `node_modules` of the
application that uses the library — it is the one runtime npm dependency. The library is
consumed as a source dependency (git or `:local/root`) by another shadow-cljs Node app
whose `shadow-cljs.edn` takes its classpath from `deps.edn`; the library's own `deps.edn`
declares ClojureScript, core.async and promesa.

```clojure
;; deps.edn of the consumer
{:deps {io.github.borr3t/tensorflow-cljs {:local/root "../tensorflow-cljs"}}}
```

```bash
npm install @tensorflow/tfjs
```

## Quick start

The quick start computes with tensors, here and now: no network, no model to download.

```bash
npm install
npx shadow-cljs compile examples && node target/examples.js
```

That runs [`dev/tensorflow_cljs/examples.cljs`](dev/tensorflow_cljs/examples.cljs), which
holds every example of this page. Its namespace form:

```clojure
(ns tensorflow-cljs.examples
  (:require [clojure.string :as str]
            [promesa.core :as p]
            [tensorflow-cljs.core :as tf]
            [tensorflow-cljs.data :as data]
            [tensorflow-cljs.layers :as layers]
            [tensorflow-cljs.models :as models]
            [tensorflow-cljs.util :as util]))
```

Two matrices, and what can be asked of them:

```clojure
(defn quick-start []
  (tf/tidy
   (fn []
     (let [a (tf/tensor [[1 2] [3 4]])
           b (tf/tensor [[5 6] [7 8]])]
       {:shape    (tf/shape a)
        :dtype    (tf/dtype a)
        :product  (tf/->vec (tf/mat-mul a b))
        :plus-one (tf/->vec (tf/add a 1))
        :mean     (tf/->vec (tf/mean a))
        :by-row   (tf/->vec (tf/sum a 1))
        :is-two?  (tf/->vec (tf/equal a 2))}))))
```

`(quick-start)` returns:

```clojure
{:shape    [2 2]
 :dtype    :float32
 :product  [[19 22] [43 50]]
 :plus-one [[2 3] [4 5]]
 :mean     2.5
 :by-row   [3 7]
 :is-two?  [[false true] [false false]]}
```

A tensor is an opaque value: `tf/tensor` makes one of ClojureScript data — the nesting
gives the shape, every number becomes a float32 unless a dtype is asked for — and
`tf/->vec` copies one back out. Tensors are not garbage-collected. `tf/tidy` runs a
function and frees every tensor made inside it, except those it returns; the quick start
returns data, so it leaves nothing behind.

## Scenarios

Each scenario is a function of `dev/tensorflow_cljs/examples.cljs`, shown with the result
it returns.

### Train a model

```clojure
(def xs [[0 0] [0 1] [1 0] [1 1]])
(def ys [[0] [1] [1] [0]])

(defn xor-model
  "A model of two layers, compiled and not yet trained. The seeds fix the
   weights it starts from, so that it trains the same way every time."
  []
  (-> (models/sequential
       [(layers/dense {:units              8
                       :activation         :tanh
                       :input-shape        [2]
                       :kernel-initializer {:type :glorot-uniform :seed 7}})
        (layers/dense {:units              1
                       :activation         :sigmoid
                       :kernel-initializer {:type :glorot-uniform :seed 11}})])
      (models/compile-model! {:optimizer {:type :adam :learning-rate 0.1}
                              :loss      :binary-crossentropy
                              :metrics   [:accuracy]})))

(defn train!
  "Trains `model` on `xs` and `ys` for `epochs` epochs. Resolves with the
   loss of every epoch."
  [model xs ys epochs]
  (let [x (tf/tensor xs)
        y (tf/tensor ys)]
    (-> (models/fit! model x y {:epochs epochs :batch-size 4 :shuffle false})
        (p/then (fn [history] (get-in history [:history :loss])))
        (p/finally (fn [_ _] (tf/dispose! [x y]))))))

(defn answers
  "What `model` answers to `xs`, every answer rounded to 0 or 1."
  [model xs]
  (tf/tidy #(tf/->vec (tf/round (models/predict model (tf/tensor xs))))))

(defn train-xor
  "Trains a model on the four cases of XOR."
  []
  (let [model (xor-model)]
    (-> (p/let [losses (train! model xs ys 150)]
          {:epochs     (count losses)
           :loss-fell? (< (last losses) (first losses))
           :accuracy   (tf/tidy #(:acc (models/evaluate model (tf/tensor xs) (tf/tensor ys))))
           :answers    (answers model xs)})
        (p/finally (fn [_ _] (models/dispose! model))))))
```

`(train-xor)` resolves with:

```clojure
{:epochs 150 :loss-fell? true :accuracy 1 :answers [[0] [1] [1] [0]]}
```

The initializers are seeded and the examples are not shuffled, so the model trains the
same way every time. `models/fit!` returns a promise: training gives the event loop its
turn between batches. `models/predict` and `models/evaluate` are synchronous.

### Save a model, load it back

```clojure
(defn save-and-load
  "Saves `model` in the directory `dir` and loads it back. Resolves with
   the files written and whether the model loaded answers as the model
   saved."
  [model dir]
  (p/let [saved  (models/save! model dir)
          loaded (models/load! dir)]
    (let [result {:files         (mapv #(str/replace % #".*/" "")
                                       [(:model-json saved) (:weights-bin saved)])
                  :same-answers? (= (answers model xs) (answers loaded xs))
                  :compiled?     (models/compiled? loaded)}]
      (models/dispose! loaded)
      result)))
```

For a trained model and a directory of yours, `(save-and-load model dir)` resolves with:

```clojure
{:files ["model.json" "weights.bin"] :same-answers? true :compiled? false}
```

The files are in the layout of TensorFlow.js: what `models/save!` writes, tfjs-node and
the TensorFlow.js converter read, and `models/load!` reads what they write. A loaded model
answers at once and is not compiled: compile it again to go on training, or save with
`{:include-optimizer true}` and the training configuration is stored with the model.
`models/load-graph!` loads a model exported by the converter, for inference.

### Feed a model from a pipeline

```clojure
(defn examples-of
  "A dataset of {:xs .. :ys ..} batches of `n` examples out of `xs` and
   `ys`, shuffled the same way every time."
  [xs ys n]
  (-> (data/from-vec (mapv (fn [x y] {:xs x :ys y}) xs ys))
      (data/shuffle 4 {:seed "xor"})
      (data/batch n)))

(defn train-from-pipeline
  "Trains a model on a dataset, two batches an epoch."
  []
  (let [model   (xor-model)
        dataset (examples-of xs ys 2)]
    (-> (p/let [batches (data/to-vec! dataset)
                shapes  (mapv (fn [{:keys [xs ys]}] [(tf/shape xs) (tf/shape ys)]) batches)
                _       (data/dispose-all! batches)
                history (models/fit-dataset! model (data/repeat dataset)
                                             {:epochs 150 :batches-per-epoch 2})]
          {:batches shapes
           :epochs  (count (:epoch history))
           :answers (answers model xs)})
        (p/finally (fn [_ _] (models/dispose! model))))))
```

`(train-from-pipeline)` resolves with:

```clojure
{:batches [[[2 2] [2 1]] [[2 2] [2 1]]]
 :epochs  150
 :answers [[0] [1] [1] [0]]}
```

`:batches` are the shapes of `:xs` and `:ys` in each batch. A dataset is a recipe, pulled
one element at a time: `data/csv` reads a file the same way, `data/generator` and
`data/func` take a function, and `data/->channel!` turns any dataset into a `core.async`
channel. `fit-dataset!` needs `:batches-per-epoch` when the dataset does not know its
size — a repeated dataset never does.

## Pure helpers and one-liners

```clojure
(tf/tidy (fn [] (tf/->vec (tf/tensor [1.9 -2.1] :int32))))
;; => [1 -2]

(tf/tidy (fn [] (tf/->vec (tf/tensor [0.1]))))
;; => [0.10000000149011612]

(tf/tidy (fn [] (tf/->vec (tf/reshape (tf/range 0 6) [2 3]))))
;; => [[0 1 2] [3 4 5]]

(tf/tidy (fn [] (tf/->vec (tf/arg-max (tf/tensor [[0.1 0.7 0.2] [0.8 0.1 0.1]]) 1))))
;; => [1 0]

(tf/tidy (fn [] (tf/shape (tf/one-hot (tf/tensor [0 2] :int32) 3))))
;; => [2 3]

(tf/tidy (fn [] (tf/->vec ((tf/grad tf/square) (tf/tensor [1 2 3])))))
;; => [2 4 6]

(tf/backend)
;; => "cpu"

(util/->config {:units 4 :kernel-initializer :glorot-uniform})
;; => {"units" 4 "kernelInitializer" "glorotUniform"}

(util/enum-value :glorot-uniform)
;; => "glorotUniform"
```

The second one-liner is the float32 nearest to 0.1, read back as a double.

## Error handling

Whatever the library throws, and whatever it rejects a promise with, is an `ex-info` whose
data carries a `:type`:

| `:type`                           | when                                                       | data                                  |
|-----------------------------------|------------------------------------------------------------|---------------------------------------|
| `:tensorflow-cljs/tf-error`       | TensorFlow.js refused the operation                        | `:code`, `:name`, `:message`, `:cause` |
| `:tensorflow-cljs/argument-error` | an argument refused before TensorFlow.js is called: an unknown dtype, loss or optimizer, rows of different lengths, a dataset of unknown size without `:batches-per-epoch` | the argument |
| `:tensorflow-cljs/io-error`       | saving or loading: a missing `model.json` or weight file, a model of the other kind | `:path` or `:dir`, `:cause` |
| `:tensorflow-cljs/backend-error`  | a backend that is not there                                | `:backend`, `:cause`                  |

The `:code` of a `:tf-error` is one of `:shape-error`, `:value-error`, `:runtime-error`,
`:assertion-error`, `:not-implemented-error`, `:attribute-error`, `:index-error`,
`:unknown`.

```clojure
(defn attempt
  "The value of `(f)`, or the failure as data."
  [f]
  (try
    {:ok? true :value (f)}
    (catch :default e
      (let [{:keys [type code]} (util/error-data e)]
        (case type
          :tensorflow-cljs/tf-error       {:ok? false :why code}
          :tensorflow-cljs/argument-error {:ok? false :why :bad-argument}
          :tensorflow-cljs/io-error       {:ok? false :why :io}
          :tensorflow-cljs/backend-error  {:ok? false :why :backend}
          (throw e))))))

(defn product
  "The matrix product of `a` and `b`, as data."
  [a b]
  (tf/tidy #(tf/->vec (tf/mat-mul (tf/tensor a) (tf/tensor b)))))
```

```clojure
(attempt (fn [] (product [[1 2] [3 4]] [[5 6] [7 8]])))
;; => {:ok? true :value [[19 22] [43 50]]}

;; a 2x2 matrix and a 1x3 matrix do not multiply
(attempt (fn [] (product [[1 2] [3 4]] [[1 2 3]])))
;; => {:ok? false :why :shape-error}

;; rows of different lengths: refused before TensorFlow.js is called
(attempt (fn [] (product [[1 2] [3]] [[1] [2]])))
;; => {:ok? false :why :bad-argument}
```

A promise rejects with the same values:

```clojure
(-> (models/load! "models/missing")
    (p/catch (fn [e] (:type (ex-data e)))))
;; => :tensorflow-cljs/io-error
```

## Features

* **Tensors** — creation from ClojureScript data with the five dtypes, random and
  constant tensors, variables and buffers; readers that copy out, synchronous and
  promise-returning.
* **The op algebra** — every op of TensorFlow.js core as a function with a kebab name:
  arithmetic, reductions, comparison, slicing and joining, linear algebra, convolution
  and pooling, image, signal, sparse and string ops.
* **Gradients and optimizers** — `grad`, `grads`, `value-and-grad`, `variable-grads`,
  `custom-grad`, and `minimize!` for a training loop of your own.
* **Layers** — every layer constructor of TensorFlow.js as a function of a config map;
  activations, initializers, regularizers and constraints named by keyword.
* **Models** — sequential and functional; `compile-model!`, `fit!`, `predict`,
  `evaluate`, `train-on-batch!`; callbacks as maps of functions, early stopping.
* **Saving and loading** — `save!` and `load!` on a directory, `save-artifacts!` and
  `load-artifacts!` in memory, `load!` from a URL, `load-graph!` for converter output.
* **Data pipelines** — sources (vectors, generators, functions, CSV files, zips),
  combinators (`batch`, `map`, `filter`, `shuffle`, `repeat`, `take`, `skip`), consumers
  as promises and as a `core.async` channel.
* **Memory under control** — `tidy`, `dispose!`, `keep`, `memory`, `num-tensors`;
  `models/dispose!` frees a model together with its optimizer.
* **No interop at call sites** — all JavaScript interop is in `tensorflow-cljs.core`;
  `layers`, `models` and `data` are plain ClojureScript over it.
* **Structured errors** — every failure is an `ex-info` with a `:type`.

## What the caller sets

The library decides nothing about a model. Its architecture, its optimizer and learning
rate, its loss, how long it trains and on what: all of it is what the caller passes, and
where the caller passes nothing the defaults of TensorFlow.js apply. Nothing is retried
and nothing times out — a computation runs until it is done. The options a caller reaches
for first:

| where                    | key                  | default        | meaning                                           |
|--------------------------|----------------------|----------------|---------------------------------------------------|
| `models/fit!`            | `:epochs`            | 1              | passes over the examples                          |
| `models/fit!`            | `:batch-size`        | 32             | examples per gradient step                        |
| `models/fit!`            | `:shuffle`           | true           | shuffle the examples before every epoch           |
| `models/fit!`            | `:validation-split`, `:validation-data` | none | examples held out of training           |
| `models/fit!`            | `:yield-every`       | `:auto`        | when training gives the event loop its turn       |
| `models/fit!`            | `:callbacks`         | none           | functions called as training goes, `early-stopping` |
| `models/fit!`            | `:verbose`           | 0              | 1 prints a progress bar                           |
| `models/fit-dataset!`    | `:batches-per-epoch` | the size of the dataset | required when the size is unknown        |
| `models/predict`         | `:batch-size`        | 32             | examples per forward pass                         |
| `models/save!`           | `:include-optimizer` | false          | store the training configuration with the model   |
| `data/->channel!`        | `:buffer`            | 32             | elements held for a slow consumer                 |
| a layer, `tf/random-normal` .. | `:seed`        | none           | what makes a run repeatable                       |

Memory is the caller's too: a tensor lives until `tidy` or `dispose!` frees it, a model
until `models/dispose!` does. `evaluate` frees what it made; `predict` leaves its output
to the caller.

The backend is the CPU backend of TensorFlow.js, in JavaScript: it needs no native build
and suits small models and pipelines. `@tensorflow/tfjs-node` can be required by the
application before the library and selected with `(tf/set-backend! "tensorflow")`; the
library's tests do not cover it.

## Worth knowing

* Every number is a float32 unless a dtype is given: `0.1` reads back as
  `0.10000000149011612`. `:int32` truncates fractions and wraps at 2³¹.
* TensorFlow.js does not check that the rows of nested data have the same length; the
  library does, and refuses with an `:argument-error`.
* `fit-dataset!` reads a dataset through once: for more than one epoch, pass
  `(data/repeat dataset)` with `:batches-per-epoch`.
* `:on-yield` of a callback map is called only with a numeric or `:auto` `:yield-every`.
* On the first tensor the CPU backend prints a banner that recommends tfjs-node.
  `(tf/prod-mode!)` silences it, and turns off the NaN checks of TensorFlow.js with it.

## Documentation

* [`docs/tensorflow-cljs.org`](docs/tensorflow-cljs.org) — the manual: tensors and
  dtypes, the op algebra, memory, the layer catalog, models and callbacks, saving and
  loading, data pipelines, backends, error handling, and the reference of every public
  function.
* [`docs/tensorflow-api-inventory.org`](docs/tensorflow-api-inventory.org) — TensorFlow.js
  itself: its packages, ops, layers, models, data and io, its errors and its pitfalls.

## Layout

```
src/tensorflow_cljs/
  core.cljs     tensors, readers, the op algebra, memory, backend, optimizers and gradients; all the interop
  layers.cljs   the layer catalog, the registries of activations, initializers, regularizers, constraints
  models.cljs   sequential and functional models, compile / fit / predict / evaluate, callbacks, save / load, graph models
  data.cljs     tf.data sources, combinators, consumers, the core.async channel
  util.cljs     kebab<->camel keys, config translation, dtypes, structured errors
dev/tensorflow_cljs/
  examples.cljs  the examples of this page
  smoke.cljs     a longer run over real files in a temporary directory
docs/
  tensorflow-cljs.org            the manual
  tensorflow-api-inventory.org   the API inventory
```

## Development

```bash
npm install
./run-tests.sh     # the publicity gate, then the unit suite; no network
npx shadow-cljs compile examples && node target/examples.js   # the examples of this page; no network
npx shadow-cljs compile smoke && node target/smoke.js         # a longer run: real files in a temporary directory; no network
```

The suite runs on the CPU backend and covers tensors and ops, memory accounting, the
layer catalog, models from training to saving and loading, data pipelines, and the
examples of this page: it fails when this page shows code or a result that the examples do
not produce.
