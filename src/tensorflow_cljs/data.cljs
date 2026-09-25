(ns tensorflow-cljs.data
  "tf.data pipelines under node (inventory §5): sources (`from-vec`,
   `generator`, `func`, `csv`, `zip`), combinators (`batch` `map` `filter`
   `shuffle` `repeat` `take` `skip` `prefetch` `concatenate`) and consumers
   (`to-vec!` `for-each!` `iterator!` `next!` `->batches!` and the
   core.async `->channel!`). Datasets and
   iterators are OPAQUE values; elements cross the boundary as ClojureScript
   maps (keys become keywords verbatim: {:xs .. :ys ..}), vectors, numbers,
   strings and opaque tensors. Every consumer returns a promesa promise
   (tf.data is pull-based, promise per element).

   Ownership (inventory 11.10): tensors inside `to-vec!` results belong to
   the caller — `dispose-all!` frees them.

   No JavaScript interop here — everything goes through tensorflow-cljs.core."
  (:refer-clojure :exclude [map filter shuffle repeat take array])
  (:require [cljs.core.async :as async :refer [<! alts! go go-loop]]
            [cljs.core.async.impl.protocols :as impl]
            [promesa.core :as p]
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
   ##Inf when the size was known, still nil when it was unknown:
   fit-dataset! then needs :batches-per-epoch either way)."
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

;; ---------------------------------------------------------------------------
;; core.async channel (house pattern shared with the leveldb-cljs iterators)
;; ---------------------------------------------------------------------------

(def default-channel-buffer
  "Channel buffer size of `->channel!` when :buffer is not given."
  32)

(def cancel-poll-ms
  "How long a parked put waits before re-checking whether the consumer
   closed the channel (core.async: closing a channel does not wake parked
   puts, so a producer parked on a full buffer must poll)."
  100)

(defn error?
  "True when a channel item is an error rather than a dataset element: an
   ex-info — tensorflow-cljs's own (:tensorflow-cljs/tf-error ..) or one
   thrown by consumer code inside `map` / `func` / `generator`, which passes
   through unchanged (tfjs prefixes its message with \"Error thrown while
   iterating through a dataset\")."
  [x]
  (instance? ExceptionInfo x))

(defn- promise->ch
  "Bridges a promise to a promise-chan delivering [:ok v] or [:error e]."
  [prom]
  (let [ch (async/promise-chan)]
    (-> (p/promise prom)
        (p/then (fn [v] (async/put! ch [:ok v])))
        (p/catch (fn [e] (async/put! ch [:error e]))))
    ch))

(defn- put-or-cancel!
  "Puts `v` on `ch`; yields true when delivered, false when the consumer
   closed `ch` (checked immediately and on every cancel-poll-ms tick while
   the put is parked)."
  [ch v]
  (go-loop []
    (let [[res port] (alts! [[ch v] (async/timeout cancel-poll-ms)])]
      (cond
        (identical? port ch) (boolean res)
        (impl/closed? ch) false
        :else (recur)))))

(defn ->channel!
  "core.async channel of the dataset's elements in order (batches when the
   dataset was batched), closed after the last one; opts {:buffer 32}. An
   error (iterator rejection) arrives as the LAST item — test with `error?`
   — and the channel closes. Closing the channel stops the producer: the
   element that could not be delivered is disposed, elements already in
   the buffer can still be taken (drain them, then `dispose-all!`). Elements
   taken from the channel belong to the consumer. Elements are pulled one at
   a time (tf.data is promise-per-element), so the buffer only smooths
   bursts."
  ([ds] (->channel! ds nil))
  ([ds {:keys [buffer] :or {buffer default-channel-buffer}}]
   (let [ch (async/chan buffer)]
     (go
       (let [[tag it] (<! (promise->ch (core/dataset-iterator! ds)))]
         (if (= :error tag)
           (do (<! (put-or-cancel! ch it)) (async/close! ch))
           (loop []
             (let [[tag r] (<! (promise->ch (core/iterator-next! it)))]
               (cond
                 (= :error tag) (do (<! (put-or-cancel! ch r)) (async/close! ch))
                 (:done r) (async/close! ch)
                 (nil? (:value r)) (recur)
                 (<! (put-or-cancel! ch (:value r))) (recur)
                 :else (core/dispose! (:value r))))))))
     ch)))

(defn channel->vec!
  "Drains a `->channel!` channel into a promise of a vector of elements; an
   error item rejects the promise (the elements before it are disposed)."
  [ch]
  (p/create
   (fn [resolve reject]
     (go-loop [acc []]
       (let [v (<! ch)]
         (cond
           (nil? v) (resolve acc)
           (error? v) (do (core/dispose! acc) (reject v))
           :else (recur (conj acc v))))))))

(defn dispose-all!
  "Disposes every tensor inside a materialised element vector (`to-vec!`,
   `->batches!` results); returns nil."
  [elements]
  (core/dispose! elements))
