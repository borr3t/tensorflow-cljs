(ns tensorflow-cljs.util
  "Small shared helpers: keyword case translation between kebab-case
   ClojureScript keywords and the camelCase config / option / registry names
   of TensorFlow.js (layer configs, fit args, optimizer names, dtypes), plus
   ex-info error construction and detection.

   Pure: no I/O and no JavaScript interop. Converting CLJS maps to JS config
   objects, wrapping tensors, and reading `message`/`name` off a thrown JS
   error is interop and therefore lives in the single binding namespace
   (tensorflow-cljs.binding, Phase 2); this namespace only deals in strings,
   keywords and maps."
  (:require [clojure.string :as str]))

;; ---------------------------------------------------------------------------
;; kebab-case <-> camelCase (inventory 9.4: every tfjs config key — layer
;; args, ModelFitArgs, CompileArgs, io options — is camelCase)
;; ---------------------------------------------------------------------------

(defn kebab->camel
  "\"kernel-initializer\" -> \"kernelInitializer\". Keywords render as their
   name (the namespace is dropped); strings pass through untouched (they are
   assumed to be exact upstream names already). Single-word names are
   unchanged (:units -> \"units\")."
  [k]
  (cond
    (keyword? k) (let [[head & tail] (str/split (name k) #"-")]
                   (apply str head (map str/capitalize tail)))
    (string? k) k
    :else (str k)))

(defn camel->kebab
  "\"kernelInitializer\" -> :kernel-initializer, \"batchInputShape\" ->
   :batch-input-shape. Works on strings and keywords, returns a keyword.
   Namespaced keywords are returned unchanged. Names that are already
   snake_case (\"val_loss\", \"kernel_regularizer\" as tfjs spells some
   history / weight keys) are also kebab-cased."
  [k]
  (if (and (keyword? k) (namespace k))
    k
    (-> (name k)
        (str/replace #"([a-z0-9])([A-Z])" "$1-$2")
        (str/replace "_" "-")
        str/lower-case
        keyword)))

(defn transform-keys
  "Recursively applies `f` to every map key in `x` (maps inside vectors/lists
   included). Keys listed in `passthrough` (a set) keep their subtree
   untouched — user-owned values (a custom layer config blob, a metadata
   map handed to save) must never be re-cased."
  ([f x] (transform-keys f #{} x))
  ([f passthrough x]
   (cond
     (map? x) (into (empty x)
                    (map (fn [[k v]]
                           (if (contains? passthrough k)
                             [(f k) v]
                             [(f k) (transform-keys f passthrough v)])))
                    x)
     (sequential? x) (into (empty x) (map #(transform-keys f passthrough %)) x)
     :else x)))

(def config-passthrough-keys
  "Kebab keys whose subtree is handed to tfjs verbatim: user metadata, opaque
   tensors / weights, and callback maps (their values are functions, their
   keys are exact tfjs hook names — see enum-value / callback-hook)."
  #{:metadata :user-defined-metadata :weights :tensors :inputs :outputs :callbacks})

(def enum-exceptions
  "Registry identifiers that are NOT the camelCase rendering of their kebab
   keyword (inventory 11.13): the gelu_new activation is the one snake_case
   activation name tfjs accepts."
  {:gelu-new "gelu_new"})

(defn enum-value
  "Keyword enum value -> tfjs registry string. Registry names are camelCase
   (:relu -> \"relu\", :glorot-uniform -> \"glorotUniform\", :categorical-crossentropy
   -> \"categoricalCrossentropy\", :channels-last -> \"channelsLast\");
   non-keywords untouched. Padding modes and dtypes are single words and thus
   unchanged (:same, :valid, :float32). enum-exceptions (:gelu-new ->
   \"gelu_new\") are honoured."
  [v]
  (if (keyword? v)
    (or (get enum-exceptions v) (kebab->camel v))
    v))

(declare ->config)

(defn config-value
  "Renders a config value: keyword enums via enum-value, nested maps via
   ->config, vectors element-wise (shapes stay numeric vectors), booleans /
   numbers / strings / functions / opaque objects untouched."
  [v]
  (cond
    (keyword? v) (enum-value v)
    (map? v) (->config v)
    (sequential? v) (mapv config-value v)
    :else v))

(defn ->config
  "Kebab keyword config map -> string-keyed camelCase map, nil values dropped,
   nested config maps translated, keyword values rendered as registry
   strings, subtrees under config-passthrough-keys untouched. nil in -> nil
   out. Pure: the result is still a CLJS map; the binding namespace turns it
   into a JS object."
  [opts]
  (when (some? opts)
    (into {}
          (keep (fn [[k v]]
                  (when (some? v)
                    (let [kk (if (string? k) (keyword k) k)]
                      [(kebab->camel k)
                       (if (contains? config-passthrough-keys kk) v (config-value v))]))))
          opts)))

(defn ->kebab-keys
  "Deeply translates camelCase / snake_case string or keyword keys (as js->clj
   yields them from a History, a summary, tf.memory()) to kebab keywords;
   values are never touched."
  [x]
  (transform-keys camel->kebab x))

;; ---------------------------------------------------------------------------
;; dtypes and shapes (inventory 2.4)
;; ---------------------------------------------------------------------------

(def dtypes
  "The tfjs DataType universe as kebab keywords -> tfjs strings."
  {:float32 "float32" :int32 "int32" :bool "bool" :string "string" :complex64 "complex64"})

(defn dtype->str
  "Keyword or string dtype -> tfjs dtype string; nil -> nil (tfjs infers).
   Throws :tensorflow-cljs/argument-error for anything else."
  [dt]
  (cond
    (nil? dt) nil
    (contains? dtypes dt) (get dtypes dt)
    (and (string? dt) (contains? (set (vals dtypes)) dt)) dt
    :else (throw (ex-info (str "tensorflow-cljs: unknown dtype " (pr-str dt))
                          {:type :tensorflow-cljs/argument-error :dtype dt
                           :allowed (set (keys dtypes))}))))

(defn str->dtype
  "\"float32\" -> :float32; unknown strings become keywords as-is."
  [s]
  (if (keyword? s) s (keyword s)))

(defn shape?
  "True for a vector/list of non-negative integers (a tfjs shape)."
  [x]
  (and (sequential? x) (every? #(and (integer? %) (>= % 0)) x)))

(defn size-of-shape
  "Product of the dimensions ([] -> 1)."
  [shape]
  (reduce * 1 shape))

(defn infer-shape
  "Shape of a nested vector, read from the first element at every level
   (see `ragged?` for the consistency check). Scalars -> []."
  [x]
  (loop [x x acc []]
    (if (sequential? x)
      (recur (first x) (conj acc (count x)))
      acc)))

(defn ragged?
  "True when nested data does not have the same length at every level of
   the shape implied by its first elements ([1 [2 3]], [[1 2] [3]],
   [[1 2] [3 4 5] [6]]). tfjs 4.22 does NOT check this itself
   (TENSORLIKE_CHECK_SHAPE_CONSISTENCY defaults to false), so the wrapper
   rejects ragged input before tfjs silently mis-shapes it."
  ([x] (ragged? x (infer-shape x)))
  ([x shape]
   (if (empty? shape)
     (sequential? x)
     (or (not (sequential? x))
         (not= (first shape) (count x))
         (boolean (some #(ragged? % (rest shape)) x))))))

;; ---------------------------------------------------------------------------
;; Errors
;; ---------------------------------------------------------------------------

(def error-types
  "Every :type a tensorflow-cljs ex-info can carry."
  #{:tensorflow-cljs/tf-error :tensorflow-cljs/argument-error
    :tensorflow-cljs/io-error :tensorflow-cljs/backend-error})

(defn classify-tf-error
  "Kebab keyword for a tfjs error given its constructor `name` and `message`
   (inventory 8): :value-error (tfjs-layers ValueError), :runtime-error,
   :not-implemented-error, :attribute-error, :index-error, :assertion-error
   (tfjs-core util.assert throws plain Error whose message starts with
   \"Error in <op>\"), :shape-error (message mentions shape / rank / dtype /
   dimension, or a \"must be int32 tensor\"-style dtype demand) or :unknown."
  [name message]
  (let [msg (str message)]
    (cond
      (= name "ValueError") :value-error
      (= name "RuntimeError") :runtime-error
      (= name "NotImplementedError") :not-implemented-error
      (= name "AttributeError") :attribute-error
      (= name "IndexError") :index-error
      (re-find #"(?i)shape|rank|dtype|dimension|must be (an? )?(int32|float32|bool|string|numeric|complex64)" msg) :shape-error
      (re-find #"^Error in " msg) :assertion-error
      :else :unknown)))

(defn tf-error
  "ex-info wrapping an error thrown by tfjs. `name` is the JS error's
   constructor name (\"ValueError\", \"Error\"...), `message` its message.
   Data: {:type :tensorflow-cljs/tf-error :code <kebab keyword from
   classify-tf-error> :name .. :message .. :cause <original error> ..}."
  [name message & [data]]
  (let [code (classify-tf-error name message)]
    (ex-info (str "tensorflow-cljs " (clojure.core/name code) ": " message)
             (merge {:type :tensorflow-cljs/tf-error
                     :code code
                     :name name
                     :message message}
                    data))))

(defn argument-error
  "ex-info for an invalid argument detected before any tfjs call."
  [message & [data]]
  (ex-info (str "tensorflow-cljs: " message)
           (merge {:type :tensorflow-cljs/argument-error} data)))

(defn io-error
  "ex-info for a model save/load failure (missing file, bad model.json,
   weight manifest mismatch, unsupported scheme)."
  [message & [data]]
  (ex-info (str "tensorflow-cljs io: " message)
           (merge {:type :tensorflow-cljs/io-error} data)))

(defn backend-error
  "ex-info for a backend that is unavailable / failed to initialise."
  [message & [data]]
  (ex-info (str "tensorflow-cljs backend: " message)
           (merge {:type :tensorflow-cljs/backend-error} data)))

(defn error-data
  "Returns the structured data of a tensorflow-cljs error, or nil when `e` is
   not one."
  [e]
  (when-let [d (ex-data e)]
    (when (contains? error-types (:type d))
      d)))

(defn error-type
  "One of the tensorflow-cljs error :type keywords (see error-types) or nil."
  [e]
  (:type (error-data e)))

(defn error-code
  "The kebab classification (:value-error, :shape-error, ...) of a
   :tensorflow-cljs/tf-error, nil otherwise."
  [e]
  (when (= :tensorflow-cljs/tf-error (error-type e))
    (:code (error-data e))))

(defn shape-error?
  "True when `e` is a tfjs error about shape / rank / dtype mismatch."
  [e]
  (= :shape-error (error-code e)))
