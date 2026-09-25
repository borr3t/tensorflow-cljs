(ns tensorflow-cljs.data
  "tf.data pipelines under node (inventory §5): sources (`from-vec`,
   `generator`, `func`, `csv`, `zip`), combinators (`batch` `map` `filter`
   `shuffle` `repeat` `take` `skip` `prefetch` `concatenate`) and consumers
   (`to-vec!` `for-each!` `iterator!` `next!` `->batches!`). Datasets and
   iterators are OPAQUE values; elements cross the boundary as ClojureScript
   maps (keys become keywords verbatim: {:xs .. :ys ..}), vectors, numbers,
   strings and opaque tensors. Every consumer returns a promesa promise
   (tf.data is pull-based, promise per element).

   Ownership (inventory 11.10): tensors inside `to-vec!` results belong to
   the caller — `dispose-all!` frees them.

   No JavaScript interop here — everything goes through tensorflow-cljs.core."
  (:refer-clojure :exclude [map filter shuffle repeat take array])
  (:require [promesa.core :as p]
            [tensorflow-cljs.core :as core]
            [tensorflow-cljs.util :as util]))

(def dataset? "True for an opaque dataset." core/dataset?)

;; ---------------------------------------------------------------------------
;; Sources
;; ---------------------------------------------------------------------------

(defn from-vec
  "Dataset over a vector of elements: numbers, strings, tensors, vectors or
   maps such as {:xs [1 2] :ys 1}; size = element count."
  [items]
  (core/dataset-array items))

(def array "Alias of `from-vec` (tf.data.array)." from-vec)

(defn generator
  "Dataset over the (possibly lazy / infinite) seq returned by calling the
   zero-argument fn `f` afresh on every iteration; size unknown (nil)."
  [f]
  (core/dataset-generator f))

(defn func
  "Dataset pulling elements from the zero-argument fn `f` until it returns
   nil (`f` may return a promise); size unknown (nil)."
  [f]
  (core/dataset-func f))

(defn csv
  "CSV dataset from a local file path (\"file://\" is added) or http(s) URL;
   config {:has-header true :column-names [..] :column-configs {\"label\"
   {:is-label true :dtype :int32 :required true :default 0}} :configured-columns-only
   false :delimiter \",\" :delim-whitespace false}. Column names are strings
   (keywords would be camelised). Rows are {:xs {:col v ..} :ys {:label v}}
   when a label column is configured, else flat {:col v ..}."
  ([source] (csv source nil))
  ([source config]
   (let [cfg (cond-> (or config {})
               (:column-configs config)
               (assoc :column-configs (into {} (clojure.core/map (fn [[k v]] [(if (keyword? k) (name k) k) (util/->config v)]))
                                            (:column-configs config))))]
     (core/dataset-csv source cfg))))

(defn zip
  "Dataset zipping a map {:a ds :b ds} (elements {:a .. :b ..}) or a vector
   [ds ds] (elements [.. ..]) of datasets."
  [container]
  (core/dataset-zip container))

;; ---------------------------------------------------------------------------
;; Combinators (pure: return a new dataset)
;; ---------------------------------------------------------------------------

(defn batch
  "Groups `n` consecutive elements into one, stacking column-wise: elements
   {:xs [2] :ys 1} become {:xs tensor[n 2] :ys tensor[n]}; `small-last?`
   (default true) keeps a final partial batch."
  ([ds n] (core/dataset-batch ds n))
  ([ds n small-last?] (core/dataset-batch ds n small-last?)))

(defn map
  "Dataset of (f element): `f` receives the element as ClojureScript data
   (tensors opaque) and returns a map / vector / number / tensor."
  [ds f]
  (core/dataset-map ds f))

(defn map!
  "Like `map` with `f` returning a promise of the new element."
  [ds f]
  (core/dataset-map-async ds f))

(defn filter
  "Keeps elements for which `pred` is truthy (size becomes nil: pass
   :batches-per-epoch to fit-dataset!)."
  [ds pred]
  (core/dataset-filter ds pred))

(defn shuffle
  "Windowed shuffle over `buffer-size` elements; opts {:seed \"string\"
   :reshuffle-each-iteration true}."
  ([ds buffer-size] (core/dataset-shuffle ds buffer-size))
  ([ds buffer-size {:keys [seed reshuffle-each-iteration]}]
   (core/dataset-shuffle ds buffer-size seed reshuffle-each-iteration)))

(defn repeat
  "Repeats the dataset `count` times; (repeat ds) repeats forever (size
   ##Inf: fit-dataset! then needs :batches-per-epoch)."
  ([ds] (core/dataset-repeat ds nil))
  ([ds count] (core/dataset-repeat ds count)))

(defn take "First `n` elements." [ds n] (core/dataset-take ds n))
(defn skip "All but the first `n` elements." [ds n] (core/dataset-skip ds n))
(defn prefetch "Prefetches `n` elements ahead." [ds n] (core/dataset-prefetch ds n))
(defn concatenate "`ds` followed by `other`." [ds other] (core/dataset-concatenate ds other))

(defn size
  "Static element count: a number, nil when unknown (generator, func,
   filter) or ##Inf after an unbounded `repeat`."
  [ds]
  (core/dataset-size ds))

(def max-buffer-size "The shuffle / prefetch buffer cap (10000)." core/dataset-max-buffer-size)

;; ---------------------------------------------------------------------------
;; Consumers (promises)
;; ---------------------------------------------------------------------------

(defn to-vec!
  "Promise of a vector of every element (materialises the whole dataset —
   never call on an unbounded one). Tensors inside belong to the caller."
  [ds]
  (core/dataset-to-vec! ds))

(defn for-each!
  "Promise resolving (to nil) after `f` ran on every element in order; `f`
   may return a promise, which is awaited."
  [ds f]
  (p/then (core/dataset-for-each! ds f) (constantly nil)))

(defn iterator!
  "Promise of an opaque iterator; step it with `next!`."
  [ds]
  (core/dataset-iterator! ds))

(defn next!
  "Promise of {:value element :done false} or {:value nil :done true}."
  [iterator]
  (core/iterator-next! iterator))

(defn ->batches!
  "Promise of a vector of {:xs tensor :ys tensor} batches of size `n` — the
   dataset materialised for custom training loops (fit-dataset! consumes the
   opaque dataset directly)."
  [ds n]
  (to-vec! (batch ds n)))

(defn column-names!
  "Promise of the column-name vector of a `csv` dataset."
  [ds]
  (core/csv-column-names! ds))

(defn dispose-all!
  "Disposes every tensor inside a materialised element vector (`to-vec!`,
   `->batches!` results); returns nil."
  [elements]
  (core/dispose! elements))
