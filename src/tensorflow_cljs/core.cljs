(ns tensorflow-cljs.core
  "The tensor core of tensorflow-cljs and the ONE namespace that touches
   JavaScript: `@tensorflow/tfjs` (pure JS, CPU backend) plus node `fs` /
   `path` for the library's own model file handler. Every other namespace
   (layers, models, data) is plain ClojureScript over the functions defined
   here and only ever receives or returns opaque values.

   What lives here:
   - tensor creation from ClojureScript data (`tensor`, `scalar`, `zeros`,
     `random-normal` ...) with `:dtype` keywords (`:float32` `:int32` `:bool`
     `:string` `:complex64`);
   - the OPAQUE tensor value (the tfjs Tensor object itself, never wrapped —
     compare through `->vec`, identity is the only equality);
   - readers that COPY OUT to ClojureScript data (`->vec` / `->clj`,
     `data-sync`, `get-element`) and their promise forms (`->vec!`, `data!`);
   - the whole op algebra of tfjs-core (inventory §2.2), one function per
     tfjs op, kebab names, positional arguments mirroring tfjs, trailing
     option maps where tfjs has many optionals;
   - memory management: `tidy`, `dispose!`, `keep`, `memory`;
   - backend / environment control;
   - optimizers, gradients and `minimize!` (custom training loops);
   - the layer / model / io / data bridges consumed by the sibling namespaces.

   Numbers -> float32: EVERY ClojureScript number becomes an IEEE float32 on
   entry unless `:dtype :int32` is given (0.1 reads back as
   0.10000000149011612; integers above 2^24 lose precision). `:int32` wraps at
   2^31 and truncates fractions. There is no float64 / int64 tensor in tfjs.

   Broadcasting: every binary arithmetic / comparison / logical op follows
   numpy rules — trailing dimensions are compared right to left, a dimension
   of 1 or a missing dimension stretches, anything else throws a
   :shape-error. Bare ClojureScript numbers are accepted wherever tfjs takes a
   TensorLike (`(add t 1)`), as are nested vectors.

   Errors: every tfjs throw is re-thrown as `ex-info` (see
   tensorflow-cljs.util/tf-error, `:code` from classify-tf-error); rejected
   promises carry the same ex-info."
  (:refer-clojure :exclude [abs max min mod range keep print reverse concat rand clone])
  (:require ["@tensorflow/tfjs" :as tf]
            ["fs" :as fs]
            ["path" :as path]
            [clojure.string :as str]
            [goog.object :as gobj]
            [promesa.core :as p]
            [tensorflow-cljs.util :as util]))

;; ---------------------------------------------------------------------------
;; Error and promise plumbing
;; ---------------------------------------------------------------------------

(defn- js-error-name
  "constructor.name of a thrown JS value (the tfjs-layers error classes do
   not set .name, inventory §8)."
  [e]
  (or (some-> e (gobj/get "constructor") (gobj/get "name")) "Error"))

(defn- wrap-error
  "A tensorflow-cljs ex-info for anything thrown by tfjs. ex-info values
   (ours or the consumer's, e.g. thrown inside a callback) pass unchanged."
  [e]
  (cond
    (some? (ex-data e)) e
    (instance? js/Error e)
    (let [nm (js-error-name e) msg (gobj/get e "message")]
      (if (and (= nm "TypeError") (re-find #"(?i)fetch" (str msg)))
        (util/io-error msg {:cause e})
        (util/tf-error nm msg {:cause e})))
    :else (util/tf-error "Error" (str e) {:cause e})))

(defn- guard
  "Calls `f`, mapping any JS throw to a tensorflow-cljs ex-info."
  [f]
  (try (f) (catch :default e (throw (wrap-error e)))))

(defn- ->promise
  "A promesa promise for a JS promise whose rejection is mapped through
   wrap-error."
  [js-promise]
  (p/catch (p/promise js-promise) (fn [e] (throw (wrap-error e)))))

(defn- async-guard
  "Runs `f` (which returns a JS promise) so that BOTH a synchronous throw and
   a rejection become a rejected promesa promise."
  [f]
  (try (->promise (f))
       (catch :default e (p/rejected (wrap-error e)))))

;; ---------------------------------------------------------------------------
;; Predicates on opaque values
;; ---------------------------------------------------------------------------

(defn tensor?
  "True for an opaque tensor (a tfjs Tensor, Variable included)."
  [x]
  (instance? tf/Tensor x))

(defn variable?
  "True for a trainable variable (see `variable`)."
  [x]
  (instance? tf/Variable x))

(defn symbolic?
  "True for a symbolic tensor (an `input` or the result of applying a layer
   to symbolic tensors — the nodes of a functional model graph)."
  [x]
  (instance? tf/SymbolicTensor x))

(defn layer?
  "True for an opaque layer value."
  [x]
  (instance? (gobj/get tf/layers "Layer") x))

(defn model?
  "True for an opaque model (sequential or functional)."
  [x]
  (instance? tf/LayersModel x))

(defn sequential-model?
  "True for an opaque sequential model."
  [x]
  (instance? tf/Sequential x))

(defn optimizer?
  "True for an opaque optimizer (see `optimizer`)."
  [x]
  (instance? tf/Optimizer x))

(defn dataset?
  "True for an opaque tf.data dataset."
  [x]
  (instance? (gobj/get tf/data "Dataset") x))

(defn callback?
  "True for an opaque training callback object (e.g. `early-stopping`)."
  [x]
  (instance? tf/CustomCallback x))

(def ^:private base-callback-class
  "tfjs-layers' BaseCallback class. The umbrella does not export it (only
   CustomCallback and callbacks.earlyStopping), so it is reached through
   CustomCallback's prototype chain; EarlyStopping extends Callback extends
   BaseCallback, so one instanceof covers every callback object."
  (-> tf/CustomCallback (gobj/get "prototype") js/Object.getPrototypeOf (gobj/get "constructor")))

(defn- base-callback? [x] (instance? base-callback-class x))

;; ---------------------------------------------------------------------------
;; ClojureScript <-> JS value conversion (the DataValues boundary, §2.7)
;; ---------------------------------------------------------------------------

(defn- key-name
  "Map key -> JS property name, verbatim (no case translation)."
  [k]
  (cond (keyword? k) (name k) (string? k) k :else (str k)))

(defn- ->js-value
  "Element conversion inside containers: nil -> null (tfjs's unknown
   dimension in shapes like [null 2]), keywords -> strings, sequentials ->
   arrays, maps -> plain objects keyed by the verbatim key name, tensors and
   every other JS value untouched."
  [x]
  (cond
    (nil? x) nil
    (tensor? x) x
    (keyword? x) (name x)
    (map? x) (let [o (js-obj)]
               (doseq [[k v] x] (gobj/set o (key-name k) (->js-value v)))
               o)
    (sequential? x) (to-array (map ->js-value x))
    (set? x) (to-array (map ->js-value x))
    :else x))

(defn- ->js-container
  "ClojureScript container -> JS container for tfjs TensorLike / NamedTensorMap
   / TensorContainer arguments (see ->js-value); a top-level nil becomes
   undefined so tfjs applies its default for an omitted optional argument."
  [x]
  (if (nil? x) js/undefined (->js-value x)))

(defn- ->arg
  "Op argument -> tfjs argument: same as ->js-container (numbers, booleans,
   strings, functions, typed arrays and tfjs objects pass through)."
  [x]
  (->js-container x))

(defn- ->config-js
  "Kebab config map -> JS object with camelCase keys and registry strings
   (tensorflow-cljs.util/->config), functions and opaque values untouched."
  [m]
  (clj->js (util/->config m)))

(defn- typed-array->vec
  "Float32Array / Int32Array / Uint8Array / string[] -> flat vector."
  [ta]
  (into [] (array-seq ta)))

(defn- bools
  "1/0 -> true/false, deeply."
  [x]
  (cond (sequential? x) (mapv bools x)
        (number? x) (== 1 x)
        :else x))

(defn- js-array->clj
  "Nested JS arrays -> nested vectors (bare values stay)."
  [x]
  (if (array? x) (mapv js-array->clj (array-seq x)) x))

(defn- element->clj
  "JS TensorContainer (dataset elements, op results, artifacts info) ->
   ClojureScript: arrays -> vectors, plain objects -> maps keyed by the
   verbatim key as a keyword, tensors opaque, typed arrays -> vectors."
  [x]
  (cond
    (nil? x) nil
    (tensor? x) x
    (array? x) (mapv element->clj x)
    (or (instance? js/Float32Array x) (instance? js/Int32Array x) (instance? js/Uint8Array x))
    (typed-array->vec x)
    (object? x) (into {} (map (fn [k] [(keyword k) (element->clj (gobj/get x k))])) (js-keys x))
    :else x))

(defn- ->result
  "Op result -> ClojureScript face: tensor stays opaque, arrays of tensors
   become vectors, NamedTensorMaps ({values, indices}) become kebab-keyed maps
   of opaque tensors, everything else passes."
  [r]
  (cond
    (tensor? r) r
    (array? r) (mapv ->result r)
    (object? r) (into {} (map (fn [k] [(util/camel->kebab k) (->result (gobj/get r k))])) (js-keys r))
    :else r))

(defn- js->kebab
  "Plain JS object / array -> ClojureScript with kebab keyword keys."
  [x]
  (util/->kebab-keys (js->clj x)))

(defn- undef-if-nil [x] (if (nil? x) js/undefined x))

(defn- dtype-arg
  "Dtype keyword -> tfjs string or undefined."
  [dt]
  (undef-if-nil (util/dtype->str dt)))

(defn- shape-arg
  "Validates a shape vector before it reaches tfjs."
  [shape]
  (when-not (util/shape? shape)
    (throw (util/argument-error (str "shape must be a vector of non-negative integers, got " (pr-str shape))
                                {:shape shape})))
  (to-array shape))

(defn- data-format-arg
  "Op data-format keyword (:nhwc :nchw :ndhwc :ncdhw :nwc :ncw) -> upper-case
   tfjs string, or undefined."
  [df]
  (cond (nil? df) js/undefined
        (keyword? df) (str/upper-case (name df))
        :else df))

(defn- enum-arg
  "Keyword -> tfjs enum string (:same -> \"same\", :channels-last ->
   \"channelsLast\"), nil -> undefined, others untouched."
  [x]
  (cond (nil? x) js/undefined (keyword? x) (util/enum-value x) :else x))

(defn- collect-tensors
  "Every opaque tensor reachable inside a ClojureScript / JS container
   (maps, vectors, sets, seqs, JS arrays and plain objects)."
  [x]
  (cond
    (tensor? x) [x]
    (map? x) (mapcat collect-tensors (vals x))
    (or (sequential? x) (set? x)) (mapcat collect-tensors x)
    (array? x) (mapcat collect-tensors (array-seq x))
    (object? x) (mapcat collect-tensors (gobj/getValues x))
    :else []))

;; ---------------------------------------------------------------------------
;; Creation (§2.1)
;; ---------------------------------------------------------------------------

(defn- creation-opts
  "Second argument of the creation fns: nil, a dtype keyword, or an option
   map {:shape [..] :dtype :int32}."
  [opts]
  (cond (nil? opts) {}
        (keyword? opts) {:dtype opts}
        (map? opts) opts
        (string? opts) {:dtype opts}
        :else (throw (util/argument-error (str "expected a dtype keyword or option map, got " (pr-str opts))
                                          {:opts opts}))))

(defn- flat-count [values]
  (if (sequential? values)
    (reduce (fn [n v] (+ n (flat-count v))) 0 values)
    1))

(defn tensor
  "Opaque tensor from ClojureScript data: a number / boolean / string
   (scalar), a flat vector, or nested vectors whose nesting gives the shape.
   `opts` is a dtype keyword or {:shape [..] :dtype ..}: with :shape the
   values are taken flat and reshaped (element count must match); :dtype is
   one of :float32 :int32 :bool :string :complex64, defaulting to float32 for
   numbers (integers included — see the namespace doc), bool for booleans,
   string for strings. Typed arrays (Float32Array, Int32Array, Uint8Array)
   are accepted as flat data. Ragged nesting throws :argument-error (the
   wrapper checks; tfjs 4.22 itself would silently mis-shape it)."
  ([values] (tensor values nil))
  ([values opts]
   (let [{:keys [shape dtype]} (creation-opts opts)]
     (when (nil? values)
       (throw (util/argument-error "tensor: values must not be nil" {})))
     (when (and (some? shape) (sequential? values)
                (not= (util/size-of-shape shape) (flat-count values)))
       (throw (util/argument-error
               (str "tensor: shape " (pr-str shape) " needs " (util/size-of-shape shape)
                    " values but " (flat-count values) " were given")
               {:shape shape :count (flat-count values)})))
     (when (and (nil? shape) (sequential? values) (util/ragged? values))
       ;; tfjs 4.22 never checks nesting consistency itself
       ;; (TENSORLIKE_CHECK_SHAPE_CONSISTENCY defaults to false): [1 [2 3]]
       ;; becomes shape [2] with data [1 2], [[1 2] [3 4 5] [6]] shape [3 2].
       (throw (util/argument-error
               (str "tensor: ragged input — the nesting implies shape " (pr-str (util/infer-shape values))
                    " but not every level has that length")
               {:shape (util/infer-shape values) :count (flat-count values)})))
     (guard #(tf/tensor (->arg values)
                        (if (some? shape) (shape-arg shape) js/undefined)
                        (dtype-arg dtype))))))

(defn- ranked-tensor [f rank]
  (fn
    ([values] (guard #(f (->arg values) js/undefined js/undefined)))
    ([values opts]
     (let [{:keys [shape dtype]} (creation-opts opts)]
       (when (and (some? shape) (not= rank (count shape)))
         (throw (util/argument-error (str "expected a rank " rank " shape, got " (pr-str shape)) {:shape shape})))
       (guard #(f (->arg values) (if (some? shape) (shape-arg shape) js/undefined) (dtype-arg dtype)))))))

(def tensor1d "Rank-1 tensor (`tensor` with a rank check); opts as in `tensor`." (ranked-tensor tf/tensor1d 1))
(def tensor2d "Rank-2 tensor (`tensor` with a rank check); opts as in `tensor`." (ranked-tensor tf/tensor2d 2))
(def tensor3d "Rank-3 tensor (`tensor` with a rank check); opts as in `tensor`." (ranked-tensor tf/tensor3d 3))
(def tensor4d "Rank-4 tensor (`tensor` with a rank check); opts as in `tensor`." (ranked-tensor tf/tensor4d 4))
(def tensor5d "Rank-5 tensor (`tensor` with a rank check); opts as in `tensor`." (ranked-tensor tf/tensor5d 5))
(def tensor6d "Rank-6 tensor (`tensor` with a rank check); opts as in `tensor`." (ranked-tensor tf/tensor6d 6))

(defn scalar
  "Rank-0 tensor from a number / boolean / string; optional dtype keyword."
  ([value] (scalar value nil))
  ([value dtype] (guard #(tf/scalar value (dtype-arg dtype)))))

(defn zeros
  "Tensor of zeros with `shape`; dtype defaults to float32."
  ([shape] (zeros shape nil))
  ([shape dtype] (guard #(tf/zeros (shape-arg shape) (dtype-arg dtype)))))

(defn ones
  "Tensor of ones with `shape`; dtype defaults to float32."
  ([shape] (ones shape nil))
  ([shape dtype] (guard #(tf/ones (shape-arg shape) (dtype-arg dtype)))))

(defn fill
  "Tensor of `shape` filled with `value` (number or string)."
  ([shape value] (fill shape value nil))
  ([shape value dtype] (guard #(tf/fill (shape-arg shape) value (dtype-arg dtype)))))

(defn zeros-like "Zeros with the shape and dtype of `x`." [x] (guard #(tf/zerosLike (->arg x))))
(defn ones-like "Ones with the shape and dtype of `x`." [x] (guard #(tf/onesLike (->arg x))))

(defn eye
  "Identity matrix of `num-rows` rows; opts {:num-columns n :batch-shape [..]
   :dtype ..}."
  ([num-rows] (eye num-rows nil))
  ([num-rows {:keys [num-columns batch-shape dtype]}]
   (guard #(tf/eye num-rows (undef-if-nil num-columns)
                   (if batch-shape (shape-arg batch-shape) js/undefined)
                   (dtype-arg dtype)))))

(defn linspace
  "`num` evenly spaced float32 values from `start` to `stop` inclusive."
  [start stop num]
  (guard #(tf/linspace start stop num)))

(defn range
  "1-D tensor [start, stop) with `step` (default 1); dtype :float32 (default)
   or :int32."
  ([start stop] (range start stop nil nil))
  ([start stop step] (range start stop step nil))
  ([start stop step dtype] (guard #(tf/range start stop (undef-if-nil step) (dtype-arg dtype)))))

(defn random-normal
  "Normal samples of `shape`; opts {:mean 0 :std-dev 1 :dtype :float32 :seed n}.
   Seeds make the CPU backend reproducible (best effort, inventory 11.16)."
  ([shape] (random-normal shape nil))
  ([shape {:keys [mean std-dev dtype seed]}]
   (guard #(tf/randomNormal (shape-arg shape) (undef-if-nil mean) (undef-if-nil std-dev)
                            (dtype-arg dtype) (undef-if-nil seed)))))

(defn random-standard-normal
  "Standard-normal samples of `shape`; opts {:dtype .. :seed n}."
  ([shape] (random-standard-normal shape nil))
  ([shape {:keys [dtype seed]}]
   (guard #(tf/randomStandardNormal (shape-arg shape) (dtype-arg dtype) (undef-if-nil seed)))))

(defn truncated-normal
  "Truncated-normal samples of `shape` (values beyond 2 std devs are
   redrawn); opts as `random-normal`."
  ([shape] (truncated-normal shape nil))
  ([shape {:keys [mean std-dev dtype seed]}]
   (guard #(tf/truncatedNormal (shape-arg shape) (undef-if-nil mean) (undef-if-nil std-dev)
                               (dtype-arg dtype) (undef-if-nil seed)))))

(defn random-uniform
  "Uniform samples of `shape`; opts {:minval 0 :maxval 1 :dtype :float32 :seed n}."
  ([shape] (random-uniform shape nil))
  ([shape {:keys [minval maxval dtype seed]}]
   (guard #(tf/randomUniform (shape-arg shape) (undef-if-nil minval) (undef-if-nil maxval)
                             (dtype-arg dtype) (undef-if-nil seed)))))

(defn random-uniform-int
  "int32 uniform samples in [minval, maxval) of `shape`; optional seed."
  ([shape minval maxval] (random-uniform-int shape minval maxval nil))
  ([shape minval maxval seed]
   (guard #(tf/randomUniformInt (shape-arg shape) minval maxval (undef-if-nil seed)))))

(defn random-gamma
  "Gamma samples of `shape` with shape parameter `alpha`; opts {:beta 1
   :dtype :float32 :seed n}."
  ([shape alpha] (random-gamma shape alpha nil))
  ([shape alpha {:keys [beta dtype seed]}]
   (guard #(tf/randomGamma (shape-arg shape) alpha (undef-if-nil beta) (dtype-arg dtype) (undef-if-nil seed)))))

(defn rand
  "Tensor of `shape` whose elements come from calling the zero-argument
   ClojureScript fn `f`; optional dtype."
  ([shape f] (rand shape f nil))
  ([shape f dtype] (guard #(tf/rand (shape-arg shape) f (dtype-arg dtype)))))

(defn multinomial
  "int32 samples (`num-samples` per row) from `logits` (1-D or 2-D); opts
   {:seed n :normalized false} (:normalized true means logits are
   probabilities)."
  ([logits num-samples] (multinomial logits num-samples nil))
  ([logits num-samples {:keys [seed normalized]}]
   (guard #(tf/multinomial (->arg logits) num-samples (undef-if-nil seed) (undef-if-nil normalized)))))

(defn one-hot
  "One-hot encoding of an int32 `indices` tensor (or vector of ints, cast to
   int32 here) with `depth` classes; opts {:on-value 1 :off-value 0 :dtype
   :int32}. Float indices throw :shape-error, as in tfjs."
  ([indices depth] (one-hot indices depth nil))
  ([indices depth {:keys [on-value off-value dtype]}]
   (let [idx (if (tensor? indices) indices (tf/tensor (->arg indices) js/undefined "int32"))]
     (guard #(tf/oneHot idx depth (undef-if-nil on-value) (undef-if-nil off-value) (dtype-arg dtype))))))

(defn complex
  "complex64 tensor from real and imaginary float32 parts (tensors or
   vectors). Read back with `real` / `imag`; `->vec` yields the interleaved
   [re im re im ..] flat data (inventory 11.7)."
  [real imag]
  (guard #(tf/complex (->arg real) (->arg imag))))

(defn real "Real part of a complex64 tensor (float32)." [x] (guard #(tf/real (->arg x))))
(defn imag "Imaginary part of a complex64 tensor (float32)." [x] (guard #(tf/imag (->arg x))))

(defn variable
  "Trainable variable initialised from tensor (or data) `init`; opts
   {:trainable true :name \"w\" :dtype ..}. Variables are NOT freed by `tidy`;
   use `dispose!` or `dispose-variables!`."
  ([init] (variable init nil))
  ([init {:keys [trainable name dtype]}]
   (guard #(tf/variable (if (tensor? init) init (tf/tensor (->arg init)))
                        (undef-if-nil trainable) (undef-if-nil name) (dtype-arg dtype)))))

(defn assign!
  "Assigns tensor / data `value` (same shape) to variable `v`; returns `v`."
  [v value]
  (guard #(js-invoke v "assign" (if (tensor? value) value (tf/tensor (->arg value)))))
  v)

(defn clone "Copy of tensor `x` (new memory)." [x] (guard #(tf/clone (->arg x))))

;; ---------------------------------------------------------------------------
;; Reading out (§2.3, §2.5): every reader COPIES to ClojureScript data
;; ---------------------------------------------------------------------------

(defn shape "Shape vector of a tensor (or symbolic tensor; nil = unknown batch dim)." [t] (js-array->clj (gobj/get t "shape")))
(defn dtype "Dtype keyword of a tensor (:float32 :int32 :bool :string :complex64)." [t] (util/str->dtype (gobj/get t "dtype")))
(defn rank "Rank (number of dimensions)." [t] (gobj/get t "rank"))
(defn size "Element count." [t] (gobj/get t "size"))
(defn strides "Row-major strides in tfjs's convention: rank-1 entries, the trailing 1 omitted ([2 3 4] -> [12 4])." [t] (js-array->clj (gobj/get t "strides")))
(defn disposed? "True once `dispose!` freed the tensor." [t] (gobj/get t "isDisposed"))

(defn- read-nested
  "arraySync-style nested JS arrays -> ClojureScript, with the bool 1/0 ->
   boolean pass (§9.5)."
  [t nested]
  (let [v (js-array->clj nested)]
    (if (= "bool" (gobj/get t "dtype")) (bools v) v)))

(defn ->vec
  "Nested vectors in the tensor's shape (a scalar gives the bare value):
   numbers for float32 / int32, true / false for bool, strings for string,
   the flat interleaved [re im ..] vector for complex64. Synchronous (a plain
   memory read on the CPU backend)."
  [t]
  (guard #(read-nested t (js-invoke t "arraySync"))))

(def ->clj "Alias of `->vec`: tensor -> ClojureScript data." ->vec)

(defn ->vec!
  "Promise of `->vec` (use when a backend may need an async transfer)."
  [t]
  (-> (async-guard #(js-invoke t "array"))
      (p/then (fn [nested] (read-nested t nested)))))

(defn data-sync
  "Flat vector of the tensor's values in row-major order (bool -> true /
   false, string -> strings, complex64 interleaved)."
  [t]
  (guard #(let [v (typed-array->vec (js-invoke t "dataSync"))]
            (if (= "bool" (gobj/get t "dtype")) (bools v) v))))

(defn data!
  "Promise of `data-sync`."
  [t]
  (-> (async-guard #(js-invoke t "data"))
      (p/then (fn [ta] (let [v (typed-array->vec ta)]
                         (if (= "bool" (gobj/get t "dtype")) (bools v) v))))))

(defn get-element
  "Single element at the given indices, e.g. (get-element t 1 2) for a
   matrix; bool tensors yield true / false."
  [t & indices]
  (guard #(let [v (apply js-invoke (js-invoke t "bufferSync") "get" indices)]
            (if (= "bool" (gobj/get t "dtype")) (== 1 v) v))))

(defn bytes!
  "Promise of the raw bytes: a vector of byte values for numeric tensors, a
   vector of per-element UTF-8 byte vectors for string tensors."
  [t]
  (-> (async-guard #(js-invoke t "bytes"))
      (p/then (fn [b] (if (array? b) (mapv typed-array->vec b) (typed-array->vec b))))))

(defn ->string
  "tfjs's own rendering of a tensor (\"Tensor\\n    [[1, 2],\\n     [3, 4]]\");
   opts {:verbose true} adds dtype / shape / rank."
  ([t] (->string t nil))
  ([t {:keys [verbose]}] (guard #(js-invoke t "toString" (boolean verbose)))))

(defn print
  "Prints `->string` of the tensor via console.log; returns nil."
  ([t] (print t nil))
  ([t opts] (js/console.log (->string t opts)) nil))

(defn ->buffer
  "Mutable TensorBuffer (opaque) of the tensor — use `buffer-get`,
   `buffer-set!` and `buffer->tensor`. Buffer interop is confined here."
  [t]
  (guard #(js-invoke t "bufferSync")))

(defn buffer
  "Fresh mutable buffer of `shape` (opaque); opts {:dtype :float32 :values
   flat-vector}. Fill with `buffer-set!`, then `buffer->tensor`."
  ([shape] (buffer shape nil))
  ([shape {:keys [dtype values]}]
   (guard #(tf/buffer (shape-arg shape) (dtype-arg dtype)
                      (if (some? values) (->arg values) js/undefined)))))

(defn buffer-get "Element of a buffer at `indices`." [b & indices]
  (apply js-invoke b "get" indices))

(defn buffer-set!
  "Sets element `value` at `indices` of buffer `b`; returns `b`."
  [b value & indices]
  (apply js-invoke b "set" value indices)
  b)

(defn buffer->tensor "Immutable tensor snapshot of a buffer." [b] (guard #(js-invoke b "toTensor")))

;; ---------------------------------------------------------------------------
;; Memory (§2.6)
;; ---------------------------------------------------------------------------

(defn tidy
  "Runs the zero-argument fn `f` in a tfjs scope: every tensor created inside
   is disposed at the end EXCEPT those reachable from the return value, which
   may be a tensor or any ClojureScript / JS container of tensors (map,
   vector, set, seq, nested). House position for inventory 11.4: the return
   value's tensors are collected by a ClojureScript walk and handed to tfjs
   as a JS array before the scope closes, so tfjs never walks persistent
   structures. `f` must be synchronous (an async fn leaks). Returns `f`'s
   value. Optional `name` labels the scope for profiling."
  ([f] (tidy nil f))
  ([name f]
   (when-not (fn? f)
     (throw (util/argument-error "tidy expects a zero-argument function" {:f f})))
   (let [result (volatile! nil)
         body (fn []
                (let [r (f)]
                  (vreset! result r)
                  (to-array (collect-tensors r))))]
     (guard #(if name (tf/tidy name body) (tf/tidy body)))
     @result)))

(defn keep
  "Exempts tensor `t` from the enclosing `tidy`; returns `t`."
  [t]
  (guard #(tf/keep t)))

(defn dispose!
  "Frees a tensor, or every tensor reachable inside a ClojureScript / JS
   container (map / vector / set / nested). Using a disposed tensor throws a
   :tensorflow-cljs/tf-error (\"Tensor is disposed.\"). Returns nil."
  [x]
  (guard #(tf/dispose (to-array (collect-tensors x))))
  nil)

(defn dispose-variables!
  "Disposes every variable, model weights included. Returns nil."
  []
  (tf/disposeVariables) nil)

(defn memory
  "Memory report as a kebab map: {:num-tensors n :num-data-buffers n
   :num-bytes n :unreliable bool :reasons [..]} (on the CPU backend
   :unreliable is always true — the count is an upper bound)."
  []
  (js->kebab (tf/memory)))

(defn num-tensors
  "Live tensor count (the leak detector of the test suite)."
  []
  (gobj/get (tf/memory) "numTensors"))

(defn profile!
  "Promise of the kebab profile info {:new-bytes :new-tensors :peak-bytes
   :kernels [..] :result ..} of running the zero-argument fn `f`."
  [f]
  (-> (async-guard #(tf/profile f))
      (p/then (fn [info] (assoc (js->kebab (js/Object.assign #js {} info :result nil))
                                :result (->result (gobj/get info "result")))))))

(defn time!
  "Promise of {:kernel-ms n :wall-ms n} for running the zero-argument fn `f`."
  [f]
  (-> (async-guard #(tf/time f)) (p/then js->kebab)))

;; ---------------------------------------------------------------------------
;; Backend / environment (§7)
;; ---------------------------------------------------------------------------

(defn backend
  "Name of the active backend (\"cpu\" under node) or nil before
   initialisation (call `ready!` or run any op first; inventory 11.2)."
  []
  (tf/getBackend))

(defn ready!
  "Promise resolving once the backend is initialised."
  []
  (async-guard #(tf/ready)))

(defn set-backend!
  "Promise of true after switching to backend `name` (\"cpu\"; \"tensorflow\"
   when the consumer required @tensorflow/tfjs-node before this namespace;
   \"webgl\" never registers under node). Rejects with
   :tensorflow-cljs/backend-error when the backend is unknown."
  [name]
  (-> (async-guard #(tf/setBackend (key-name name)))
      (p/catch (fn [e] (throw (util/backend-error (str "cannot select backend " (pr-str name) ": " (ex-message e))
                                                  {:backend name :cause e}))))))

(defn find-backend
  "True when backend `name` is registered."
  [name]
  (some? (tf/findBackend (key-name name))))

(defn env-flag
  "Value of a tfjs environment flag (:is-node, :prod, :debug, ..) — keyword
   names are upper-snake-cased (:is-node -> \"IS_NODE\"), strings verbatim."
  [flag]
  (let [n (if (keyword? flag) (str/upper-case (str/replace (name flag) "-" "_")) flag)]
    (guard #(js-invoke (tf/env) "get" n))))

(defn env-flags
  "Every registered environment flag as a kebab map."
  []
  (into {} (map (fn [[k v]] [(keyword (str/lower-case (str/replace k "_" "-"))) v]))
        (js->clj (js-invoke (tf/env) "getFlags"))))

(defn prod-mode!
  "Enables tfjs production mode: silences the node banner, skips the
   TensorLike shape-consistency check and NaN checks (inventory 11.5). Not
   called by default."
  []
  (tf/enableProdMode) nil)

(defn debug-mode!
  "Enables tfjs debug mode (logs every kernel; slow)."
  []
  (tf/enableDebugMode) nil)

(defn disable-deprecation-warnings!
  "Silences tfjs deprecation warnings."
  []
  (tf/disableDeprecationWarnings) nil)

(defn version
  "Versions of the bundled tfjs packages as a kebab map (:tfjs :tfjs-core ..)."
  []
  (into {} (map (fn [[k v]] [(keyword k) v])) (js->clj tf/version)))

;; ---------------------------------------------------------------------------
;; The op algebra (§2.2). Helpers build the trivial arities; every op is a
;; public fn with a docstring. Positional args mirror tfjs; nil optionals
;; become undefined so tfjs applies its defaults.
;; ---------------------------------------------------------------------------

(defn- call
  "Applies tfjs fn `f` to converted `args`, converting the result."
  [f & args]
  (guard #(->result (apply f (map ->arg args)))))

(defn- unary [f] (fn [x] (call f x)))
(defn- binary [f] (fn [a b] (call f a b)))
(defn- reduction [f] (fn ([x] (call f x)) ([x axis] (call f x axis)) ([x axis keep-dims] (call f x axis keep-dims))))
(defn- ns-fn [obj name] (gobj/get obj name))

;; Arithmetic (24) — broadcasting per the namespace doc
(def add "Elementwise a + b (broadcasting)." (binary tf/add))
(defn add-n "Sum of a vector of same-shaped tensors." [tensors] (call tf/addN tensors))
(def sub "Elementwise a - b (broadcasting)." (binary tf/sub))
(def mul "Elementwise a * b (broadcasting)." (binary tf/mul))
(def div "Elementwise a / b (broadcasting; int32 / int32 truncates)." (binary tf/div))
(def div-no-nan "a / b with 0 where b is 0." (binary tf/divNoNan))
(def floor-div "Elementwise floor(a / b)." (binary tf/floorDiv))
(def mod "Elementwise a mod b (sign of the divisor, as tfjs)." (binary tf/mod))
(def pow "Elementwise base ^ exp." (binary tf/pow))
(def neg "Elementwise -x." (unary tf/neg))
(def abs "Elementwise |x|." (unary tf/abs))
(def sign "Elementwise sign (-1, 0, 1)." (unary tf/sign))
(def square "Elementwise x²." (unary tf/square))
(def sqrt "Elementwise √x." (unary tf/sqrt))
(def rsqrt "Elementwise 1 / √x." (unary tf/rsqrt))
(def reciprocal "Elementwise 1 / x." (unary tf/reciprocal))
(def squared-difference "Elementwise (a - b)²." (binary tf/squaredDifference))
(def maximum "Elementwise max(a, b)." (binary tf/maximum))
(def minimum "Elementwise min(a, b)." (binary tf/minimum))
(def atan2 "Elementwise atan2(a, b)." (binary tf/atan2))
(defn clip-by-value "Clamps every element into [lo, hi]." [x lo hi] (call tf/clipByValue x lo hi))
(def round "Elementwise round half to even." (unary tf/round))
(def floor "Elementwise floor." (unary tf/floor))
(def ceil "Elementwise ceil." (unary tf/ceil))

;; Elementwise math (22)
(def exp "Elementwise eˣ." (unary tf/exp))
(def expm1 "Elementwise eˣ - 1." (unary tf/expm1))
(def log "Elementwise natural log." (unary tf/log))
(def log1p "Elementwise log(1 + x)." (unary tf/log1p))
(def sin "Elementwise sin." (unary tf/sin))
(def cos "Elementwise cos." (unary tf/cos))
(def tan "Elementwise tan." (unary tf/tan))
(def asin "Elementwise asin." (unary tf/asin))
(def acos "Elementwise acos." (unary tf/acos))
(def atan "Elementwise atan." (unary tf/atan))
(def sinh "Elementwise sinh." (unary tf/sinh))
(def cosh "Elementwise cosh." (unary tf/cosh))
(def tanh "Elementwise tanh." (unary tf/tanh))
(def asinh "Elementwise asinh." (unary tf/asinh))
(def acosh "Elementwise acosh." (unary tf/acosh))
(def atanh "Elementwise atanh." (unary tf/atanh))
(def erf "Elementwise Gauss error function." (unary tf/erf))
(def is-finite "Bool tensor: element is finite." (unary tf/isFinite))
(def is-inf "Bool tensor: element is ±Infinity." (unary tf/isInf))
(def is-nan "Bool tensor: element is NaN." (unary tf/isNaN))
(defn step "Elementwise 0 for x <= 0 else 1; `alpha` (default 0) is the value at x = 0." ([x] (call tf/step x)) ([x alpha] (call tf/step x alpha)))
(def bitwise-and "Elementwise bitwise AND of two int32 tensors." (binary tf/bitwiseAnd))

;; Activations (12)
(def relu "Elementwise max(x, 0)." (unary tf/relu))
(def relu6 "Elementwise min(max(x, 0), 6)." (unary tf/relu6))
(def elu "Elementwise exponential linear unit." (unary tf/elu))
(def selu "Elementwise scaled ELU." (unary tf/selu))
(def prelu "Parametric ReLU: x if x > 0 else alpha * x (alpha a tensor)." (binary tf/prelu))
(defn leaky-relu "Elementwise leaky ReLU with slope `alpha` (default 0.2)." ([x] (call tf/leakyRelu x)) ([x alpha] (call tf/leakyRelu x alpha)))
(def sigmoid "Elementwise logistic sigmoid." (unary tf/sigmoid))
(def log-sigmoid "Elementwise log(sigmoid(x))." (unary tf/logSigmoid))
(def softplus "Elementwise log(1 + eˣ)." (unary tf/softplus))
(defn softmax "Softmax along `dim` (default -1, the last axis)." ([x] (call tf/softmax x)) ([x dim] (call tf/softmax x dim)))
(defn log-softmax "log(softmax) along `axis` (default -1)." ([x] (call tf/logSoftmax x)) ([x axis] (call tf/logSoftmax x axis)))
(defn dropout
  "Randomly zeroes elements with probability `rate`, scaling the rest by
   1/(1-rate); opts {:noise-shape [..] :seed n}."
  ([x rate] (dropout x rate nil))
  ([x rate {:keys [noise-shape seed]}] (call tf/dropout x rate noise-shape seed)))

;; Reductions (15): (op x), (op x axis), (op x axis keep-dims); axis may be a vector
(def sum "Sum over `axis` (all axes by default)." (reduction tf/sum))
(def mean "Mean over `axis` (all axes by default)." (reduction tf/mean))
(def prod "Product over `axis`." (reduction tf/prod))
(def max "Maximum over `axis`." (reduction tf/max))
(def min "Minimum over `axis`." (reduction tf/min))
(defn arg-max "int32 indices of the maxima along `axis` (default 0)." ([x] (call tf/argMax x)) ([x axis] (call tf/argMax x axis)))
(defn arg-min "int32 indices of the minima along `axis` (default 0)." ([x] (call tf/argMin x)) ([x axis] (call tf/argMin x axis)))
(def all "Logical AND over `axis` of a bool tensor." (reduction tf/all))
(def any "Logical OR over `axis` of a bool tensor." (reduction tf/any))
(def log-sum-exp "log(Σ eˣ) over `axis`." (reduction tf/logSumExp))
(def euclidean-norm "√(Σ x²) over `axis`." (reduction tf/euclideanNorm))
(defn norm
  "Matrix / vector norm; `ord` 1, 2, Infinity, -Infinity, :euclidean or
   :fro (default :euclidean), `axis` number or vector, `keep-dims`."
  ([x] (call tf/norm x))
  ([x ord] (call tf/norm x ord))
  ([x ord axis] (call tf/norm x ord axis))
  ([x ord axis keep-dims] (call tf/norm x ord axis keep-dims)))
(defn moments
  "{:mean t :variance t} over `axis` (all axes by default)."
  ([x] (call tf/moments x)) ([x axis] (call tf/moments x axis)) ([x axis keep-dims] (call tf/moments x axis keep-dims)))
(defn cumsum
  "Cumulative sum along `axis` (default 0); opts {:exclusive false :reverse false}."
  ([x] (cumsum x nil nil)) ([x axis] (cumsum x axis nil))
  ([x axis {:keys [exclusive reverse]}] (call tf/cumsum x axis exclusive reverse)))
(defn cumprod
  "Cumulative product along `axis` (default 0); opts {:exclusive false :reverse false}."
  ([x] (cumprod x nil nil)) ([x axis] (cumprod x axis nil))
  ([x axis {:keys [exclusive reverse]}] (call tf/cumprod x axis exclusive reverse)))

;; Comparison / logical (12): bool results
(def equal "Elementwise a == b (bool)." (binary tf/equal))
(def not-equal "Elementwise a != b (bool)." (binary tf/notEqual))
(def greater "Elementwise a > b (bool)." (binary tf/greater))
(def greater-equal "Elementwise a >= b (bool)." (binary tf/greaterEqual))
(def less "Elementwise a < b (bool)." (binary tf/less))
(def less-equal "Elementwise a <= b (bool)." (binary tf/lessEqual))
(def logical-and "Elementwise AND of bool tensors." (binary tf/logicalAnd))
(def logical-or "Elementwise OR of bool tensors." (binary tf/logicalOr))
(def logical-not "Elementwise NOT of a bool tensor." (unary tf/logicalNot))
(def logical-xor "Elementwise XOR of bool tensors." (binary tf/logicalXor))
(defn where "Elementwise: `a` where `condition` (bool) is true, else `b`." [condition a b] (call tf/where condition a b))
(defn where!
  "Promise of the int32 [n rank] coordinates of the true elements of a bool
   tensor."
  [condition]
  (async-guard #(tf/whereAsync (->arg condition))))

;; Shape / geometry (44)
(defn reshape "New view of `x` with `shape` (one -1 allowed; element count must match)." [x shape] (call tf/reshape x shape))
(defn transpose
  "Permutes dimensions (`perm`, default reversed); `conjugate` for complex64."
  ([x] (call tf/transpose x)) ([x perm] (call tf/transpose x perm)) ([x perm conjugate] (call tf/transpose x perm conjugate)))
(defn expand-dims "Inserts a size-1 dimension at `axis` (default 0)." ([x] (call tf/expandDims x)) ([x axis] (call tf/expandDims x axis)))
(defn squeeze "Removes size-1 dimensions (all, or those listed in the `axis` vector)." ([x] (call tf/squeeze x)) ([x axis] (call tf/squeeze x axis)))
(defn concat "Concatenates a vector of tensors along `axis` (default 0)." ([tensors] (call tf/concat tensors)) ([tensors axis] (call tf/concat tensors axis)))
(defn stack "Stacks a vector of same-shaped tensors into a new `axis` (default 0)." ([tensors] (call tf/stack tensors)) ([tensors axis] (call tf/stack tensors axis)))
(defn unstack "Vector of tensors from unpacking `axis` (default 0)." ([x] (call tf/unstack x)) ([x axis] (call tf/unstack x axis)))
(defn split
  "Vector of tensors: `num-or-sizes` equal splits (number) or a vector of
   sizes, along `axis` (default 0)."
  ([x num-or-sizes] (call tf/split x num-or-sizes)) ([x num-or-sizes axis] (call tf/split x num-or-sizes axis)))
(defn slice
  "Sub-tensor starting at `begin` (number or vector) with `size` (number or
   vector, -1 = to the end; default to the end)."
  ([x begin] (call tf/slice x begin)) ([x begin size] (call tf/slice x begin size)))
(defn strided-slice
  "Numpy-style strided slice from `begin` to `end` (vectors) with `strides`;
   opts {:begin-mask :end-mask :ellipsis-mask :new-axis-mask :shrink-axis-mask}
   are bit masks."
  ([x begin end] (strided-slice x begin end nil nil))
  ([x begin end strides] (strided-slice x begin end strides nil))
  ([x begin end strides {:keys [begin-mask end-mask ellipsis-mask new-axis-mask shrink-axis-mask]}]
   (call tf/stridedSlice x begin end strides begin-mask end-mask ellipsis-mask new-axis-mask shrink-axis-mask)))
(defn gather
  "Slices of `x` at int32 `indices` along `axis` (default 0); `batch-dims` for
   batched gathers."
  ([x indices] (call tf/gather x indices)) ([x indices axis] (call tf/gather x indices axis))
  ([x indices axis batch-dims] (call tf/gather x indices axis batch-dims)))
(defn gather-nd "Gathers slices of `x` at multi-dimensional int32 `indices`." [x indices] (call tf/gatherND x indices))
(defn scatter-nd "Tensor of `shape` with `updates` scattered at `indices` (zeros elsewhere)." [indices updates shape] (call tf/scatterND indices updates shape))
(defn tensor-scatter-update "Copy of `tensor` with `updates` written at `indices`." [tensor indices updates] (call tf/tensorScatterUpdate tensor indices updates))
(defn reverse "Reverses `x` along `axis` (number or vector; all axes by default)." ([x] (call tf/reverse x)) ([x axis] (call tf/reverse x axis)))
(defn tile "Repeats `x` `reps` times per dimension (vector)." [x reps] (call tf/tile x reps))
(defn pad
  "Pads with `constant-value` (default 0): `paddings` is [[before after] ..] per dimension."
  ([x paddings] (call tf/pad x paddings)) ([x paddings constant-value] (call tf/pad x paddings constant-value)))
(defn mirror-pad "Pads by mirroring; `mode` :reflect or :symmetric." [x paddings mode] (call tf/mirrorPad x paddings (enum-arg mode)))
(defn broadcast-to "Broadcasts `x` to `shape`." [x shape] (call tf/broadcastTo x shape))
(defn broadcast-args "int32 broadcast shape of two shape tensors / vectors." [s0 s1] (call tf/broadcastArgs s0 s1))
(defn batch-to-space-nd "Inverse of space-to-batch-nd; `crops` [[a b] ..]." [x block-shape crops] (call tf/batchToSpaceND x block-shape crops))
(defn space-to-batch-nd "Divides spatial dims into blocks moved to the batch dim; `paddings` [[a b] ..]." [x block-shape paddings] (call tf/spaceToBatchND x block-shape paddings))
(defn depth-to-space "Rearranges depth into spatial blocks of `block-size`; opts {:data-format :nhwc}." ([x block-size] (call tf/depthToSpace x block-size)) ([x block-size {:keys [data-format]}] (call tf/depthToSpace x block-size (data-format-arg data-format))))
(defn boolean-mask!
  "Promise of the elements of `x` where bool `mask` is true (along `axis`)."
  ([x mask] (boolean-mask! x mask nil))
  ([x mask axis] (async-guard #(tf/booleanMaskAsync (->arg x) (->arg mask) (undef-if-nil axis)))))
(defn ensure-shape "Returns `x` if its shape matches `shape` (nil = any), else throws :shape-error." [x shape] (call tf/ensureShape x shape))
(defn cast "Converts to `dtype` (:float32 :int32 :bool ..); bool -> 1.0 / 0.0, float -> int32 truncates." [x dtype] (call tf/cast x (util/dtype->str dtype)))
(defn meshgrid
  "[X Y] coordinate matrices of two 1-D tensors; opts {:indexing :xy} (or :ij)."
  ([x y] (call tf/meshgrid x y))
  ([x y {:keys [indexing]}] (call tf/meshgrid x y (->config-js {:indexing indexing}))))
(defn diag "Diagonal matrix (rank 2n) from `x` (rank n)." [x] (call tf/diag x))

;; Linear algebra (8 + linalg.*)
(defn mat-mul
  "Matrix product a·b (batched over leading dims); opts {:transpose-a bool
   :transpose-b bool}. Inner dimensions must match (:shape-error)."
  ([a b] (call tf/matMul a b))
  ([a b {:keys [transpose-a transpose-b]}] (call tf/matMul a b transpose-a transpose-b)))
(def matmul "Alias of `mat-mul`." mat-mul)
(defn dot "Dot product of vectors / matrices (rank 1 or 2)." [a b] (call tf/dot a b))
(defn outer-product "Outer product of two 1-D tensors." [a b] (call tf/outerProduct a b))
(defn einsum "Einstein summation: (einsum \"ij,jk->ik\" a b)." [equation & tensors] (apply call tf/einsum equation tensors))
(defn batch-norm
  "Batch normalisation of `x` with `mean` / `variance`; opts {:offset t :scale
   t :variance-epsilon 0.001}."
  ([x mean variance] (batch-norm x mean variance nil))
  ([x mean variance {:keys [offset scale variance-epsilon]}] (call tf/batchNorm x mean variance offset scale variance-epsilon)))
(defn linalg-band-part "Keeps `num-lower` sub- and `num-upper` super-diagonals (-1 = all)." [a num-lower num-upper] (call (ns-fn tf/linalg "bandPart") a num-lower num-upper))
(defn linalg-gram-schmidt "Orthogonalises the rows of a matrix (or vector of 1-D tensors)." [xs] (call (ns-fn tf/linalg "gramSchmidt") xs))
(defn linalg-qr "QR decomposition as [q r]; `full-matrices` (default false)." ([x] (call (ns-fn tf/linalg "qr") x)) ([x full-matrices] (call (ns-fn tf/linalg "qr") x full-matrices)))

;; Convolution / pooling (15 + fused.*): trailing option maps
(defn conv1d
  "1-D convolution of `x` [batch width channels] with `filter` [width in out];
   opts {:stride 1 :pad :valid|:same|n :data-format :nwc :dilation 1
   :dim-rounding-mode :floor|:round|:ceil}."
  ([x filter] (conv1d x filter nil))
  ([x filter {:keys [stride pad data-format dilation dim-rounding-mode] :or {stride 1 pad :valid}}]
   (call tf/conv1d x filter stride (enum-arg pad) (data-format-arg data-format) dilation (enum-arg dim-rounding-mode))))
(defn conv2d
  "2-D convolution of `x` [batch h w in] (or [h w in]) with `filter` [fh fw in
   out]; opts {:strides 1|[sh sw] :pad :valid|:same|n :data-format :nhwc
   :dilations 1 :dim-rounding-mode ..}."
  ([x filter] (conv2d x filter nil))
  ([x filter {:keys [strides pad data-format dilations dim-rounding-mode] :or {strides 1 pad :valid}}]
   (call tf/conv2d x filter strides (enum-arg pad) (data-format-arg data-format) dilations (enum-arg dim-rounding-mode))))
(defn conv2d-transpose
  "Transposed 2-D convolution producing `output-shape` ([batch h w out] or
   [h w out]); opts {:strides 1 :pad :valid :dim-rounding-mode ..}."
  ([x filter output-shape] (conv2d-transpose x filter output-shape nil))
  ([x filter output-shape {:keys [strides pad dim-rounding-mode] :or {strides 1 pad :valid}}]
   (call tf/conv2dTranspose x filter output-shape strides (enum-arg pad) (enum-arg dim-rounding-mode))))
(defn conv3d
  "3-D convolution; opts {:strides 1 :pad :valid :data-format :ndhwc :dilations 1}."
  ([x filter] (conv3d x filter nil))
  ([x filter {:keys [strides pad data-format dilations] :or {strides 1 pad :valid}}]
   (call tf/conv3d x filter strides (enum-arg pad) (data-format-arg data-format) dilations)))
(defn conv3d-transpose
  "Transposed 3-D convolution to `output-shape`; opts {:strides 1 :pad :valid}."
  ([x filter output-shape] (conv3d-transpose x filter output-shape nil))
  ([x filter output-shape {:keys [strides pad] :or {strides 1 pad :valid}}]
   (call tf/conv3dTranspose x filter output-shape strides (enum-arg pad))))
(defn depthwise-conv2d
  "Depthwise 2-D convolution with `filter` [fh fw in multiplier]; opts as conv2d."
  ([x filter] (depthwise-conv2d x filter nil))
  ([x filter {:keys [strides pad data-format dilations dim-rounding-mode] :or {strides 1 pad :valid}}]
   (call tf/depthwiseConv2d x filter strides (enum-arg pad) (data-format-arg data-format) dilations (enum-arg dim-rounding-mode))))
(defn separable-conv2d
  "Depthwise then pointwise 2-D convolution; opts {:strides 1 :pad :valid
   :dilation 1 :data-format :nhwc}."
  ([x depthwise-filter pointwise-filter] (separable-conv2d x depthwise-filter pointwise-filter nil))
  ([x depthwise-filter pointwise-filter {:keys [strides pad dilation data-format] :or {strides 1 pad :valid}}]
   (call tf/separableConv2d x depthwise-filter pointwise-filter strides (enum-arg pad) dilation (data-format-arg data-format))))
(defn dilation2d
  "Grayscale morphological dilation with `filter` [fh fw depth]; opts
   {:strides 1 :pad :valid :dilations 1 :data-format :nhwc}."
  ([x filter] (dilation2d x filter nil))
  ([x filter {:keys [strides pad dilations data-format] :or {strides 1 pad :valid}}]
   (call tf/dilation2d x filter strides (enum-arg pad) dilations (data-format-arg data-format))))
(defn- pool-op [f]
  (fn
    ([x filter-size] (call f x filter-size 1 "valid"))
    ([x filter-size {:keys [strides pad dim-rounding-mode] :or {strides 1 pad :valid}}]
     (call f x filter-size strides (enum-arg pad) (enum-arg dim-rounding-mode)))))
(def avg-pool
  "2-D average pooling with `filter-size` (number or [h w]); opts {:strides 1
   :pad :valid :dim-rounding-mode ..}."
  (pool-op tf/avgPool))
(def max-pool
  "2-D max pooling with `filter-size` (number or [h w]); opts {:strides 1
   :pad :valid :dim-rounding-mode ..}."
  (pool-op tf/maxPool))
(defn- pool3d-op [f]
  (fn
    ([x filter-size] (call f x filter-size 1 "valid"))
    ([x filter-size {:keys [strides pad dim-rounding-mode data-format] :or {strides 1 pad :valid}}]
     (call f x filter-size strides (enum-arg pad) (enum-arg dim-rounding-mode) (data-format-arg data-format)))))
(def avg-pool3d "3-D average pooling; opts {:strides 1 :pad :valid :dim-rounding-mode .. :data-format :ndhwc}." (pool3d-op tf/avgPool3d))
(def max-pool3d "3-D max pooling; opts as avg-pool3d." (pool3d-op tf/maxPool3d))
(defn max-pool-with-argmax
  "{:result t :indexes t} of 2-D max pooling; opts {:strides 1 :pad :valid
   :include-batch-in-index false}."
  ([x filter-size] (max-pool-with-argmax x filter-size nil))
  ([x filter-size {:keys [strides pad include-batch-in-index] :or {strides 1 pad :valid}}]
   (call tf/maxPoolWithArgmax x filter-size strides (enum-arg pad) include-batch-in-index)))
(defn pool
  "Generic N-D pooling with `window-shape` and `pooling-type` :avg / :max;
   opts {:pad :valid :dilations 1 :strides 1 :dim-rounding-mode ..}."
  ([x window-shape pooling-type] (pool x window-shape pooling-type nil))
  ([x window-shape pooling-type {:keys [pad dilations strides dim-rounding-mode] :or {pad :valid}}]
   (call tf/pool x window-shape (enum-arg pooling-type) (enum-arg pad) dilations strides (enum-arg dim-rounding-mode))))
(defn local-response-normalization
  "LRN over the channel dim; opts {:depth-radius 5 :bias 1 :alpha 1 :beta 0.5}."
  ([x] (local-response-normalization x nil))
  ([x {:keys [depth-radius bias alpha beta]}] (call tf/localResponseNormalization x depth-radius bias alpha beta)))
(defn- fused-args [m] (->config-js (update m :activation enum-arg)))
(defn fused-conv2d
  "Fused conv2d + bias + activation: one map {:x :filter :strides :pad
   :data-format :dilations :dim-rounding-mode :bias :activation (:relu
   :relu6 :prelu :leakyrelu :sigmoid :elu :linear) :prelu-activation-weights
   :leakyrelu-alpha}."
  [args] (call (ns-fn tf/fused "conv2d") (fused-args args)))
(defn fused-depthwise-conv2d "Fused depthwise conv2d; args map as `fused-conv2d`." [args] (call (ns-fn tf/fused "depthwiseConv2d") (fused-args args)))
(defn fused-mat-mul
  "Fused matMul + bias + activation: {:a :b :transpose-a :transpose-b :bias
   :activation :prelu-activation-weights :leakyrelu-alpha}."
  [args] (call (ns-fn tf/fused "matMul") (fused-args args)))

;; Recurrent cells (2)
(defn basic-lstm-cell
  "One LSTM step: [new-c new-h] from `forget-bias` (scalar), `lstm-kernel`,
   `lstm-bias`, `data`, previous `c` and `h`."
  [forget-bias lstm-kernel lstm-bias data c h]
  (call tf/basicLSTMCell forget-bias lstm-kernel lstm-bias data c h))
(defn multi-rnn-cell
  "Stacked LSTM step: [cs hs] from a vector of cell fns (each (fn [data c h])
   -> [c h]), `data`, and vectors `c` / `h` of previous states."
  [lstm-cells data c h]
  (call tf/multiRNNCell (to-array (map (fn [f] (fn [d c h] (->js-container (f d c h)))) lstm-cells)) data c h))

;; Search / sort / sets (9)
(defn topk
  "{:values t :indices t} of the `k` largest entries along the last axis;
   `sorted` (default true)."
  ([x] (call tf/topk x)) ([x k] (call tf/topk x k)) ([x k sorted] (call tf/topk x k sorted)))
(defn unique "{:values t :indices t} of the unique elements along `axis` (default 0)." ([x] (call tf/unique x)) ([x axis] (call tf/unique x axis)))
(defn search-sorted "int32 insertion indices of `values` into `sorted-sequence`; `side` :left (default) or :right." ([s values] (call tf/searchSorted s values)) ([s values side] (call tf/searchSorted s values (enum-arg side))))
(defn lower-bound "int32 left insertion indices of `values` into `sorted-sequence`." [s values] (call tf/lowerBound s values))
(defn upper-bound "int32 right insertion indices of `values` into `sorted-sequence`." [s values] (call tf/upperBound s values))
(defn in-top-k!
  "Promise of a bool tensor: whether each target is in the top `k` (default 1)
   predictions."
  ([predictions targets] (in-top-k! predictions targets nil))
  ([predictions targets k] (async-guard #(tf/inTopKAsync (->arg predictions) (->arg targets) (undef-if-nil k)))))
(defn setdiff1d!
  "Promise of [values indices]: elements of `x` not in `y`."
  [x y] (-> (async-guard #(tf/setdiff1dAsync (->arg x) (->arg y))) (p/then ->result)))
(defn bincount "Counts of each int32 value in `x` (length `size`), `weights` (tensor or []) summed instead of 1 when given." [x weights size] (call tf/bincount x weights size))
(defn dense-bincount "Row-wise `bincount` of a 1-D / 2-D int32 tensor; `binary-output` clamps counts to 1." ([x weights size] (call tf/denseBincount x weights size)) ([x weights size binary-output] (call tf/denseBincount x weights size binary-output)))

;; Segment / sparse / ragged (5 + sparse.*)
(defn unsorted-segment-sum "Sums `x` rows by int32 `segment-ids` into `num-segments` rows." [x segment-ids num-segments] (call tf/unsortedSegmentSum x segment-ids num-segments))
(defn sparse-to-dense "Dense tensor of `output-shape` with `sparse-values` at `sparse-indices`; `default-value` elsewhere (0)." ([indices values output-shape] (call tf/sparseToDense indices values output-shape)) ([indices values output-shape default-value] (call tf/sparseToDense indices values output-shape default-value)))
(defn ragged-gather "{:output-nested-splits [..] :output-dense-values t} gather over a ragged tensor." [params-nested-splits params-dense-values indices output-ragged-rank] (call tf/raggedGather params-nested-splits params-dense-values indices output-ragged-rank))
(defn ragged-range "{:rt-nested-splits t :rt-dense-values t} of ranges [starts, limits) by deltas." [starts limits deltas] (call tf/raggedRange starts limits deltas))
(defn ragged-tensor-to-tensor "Dense tensor of `shape` from ragged `values` and row partitions (`row-partition-types` a vector of strings)." [shape values default-value row-partition-tensors row-partition-types] (call tf/raggedTensorToTensor shape values default-value row-partition-tensors row-partition-types))
(defn sparse-fill-empty-rows "{:output-indices :output-values :empty-row-indicator :reverse-index-map} filling empty rows with `default-value`." [indices values dense-shape default-value] (call (ns-fn tf/sparse "sparseFillEmptyRows") indices values dense-shape default-value))
(defn sparse-reshape "{:output-indices :output-shape} of reshaping a sparse tensor to `new-shape`." [input-indices input-shape new-shape] (call (ns-fn tf/sparse "sparseReshape") input-indices input-shape new-shape))
(defn sparse-segment-mean "Mean of `data` rows at `indices` grouped by `segment-ids`." [data indices segment-ids] (call (ns-fn tf/sparse "sparseSegmentMean") data indices segment-ids))
(defn sparse-segment-sum "Sum of `data` rows at `indices` grouped by `segment-ids`." [data indices segment-ids] (call (ns-fn tf/sparse "sparseSegmentSum") data indices segment-ids))

;; Signal / spectral (6 + signal.*)
(def fft "Fast Fourier transform of a complex64 tensor (inner dimension)." (unary tf/fft))
(def ifft "Inverse FFT of a complex64 tensor." (unary tf/ifft))
(defn rfft "FFT of a real tensor -> complex64; optional `fft-length`." ([x] (call tf/rfft x)) ([x fft-length] (call tf/rfft x fft-length)))
(def irfft "Inverse real FFT of a complex64 tensor -> float32." (unary tf/irfft))
(defn enclosing-power-of-two "Smallest power of two >= n (a number)." [n] (guard #(tf/enclosingPowerOfTwo n)))
(defn cosine-window "Generalised cosine window a - b·cos(2πn/(N-1)) of `window-length`." [window-length a b] (call tf/cosineWindow window-length a b))
(defn signal-hamming-window "Hamming window of `window-length`." [window-length] (call (ns-fn tf/signal "hammingWindow") window-length))
(defn signal-hann-window "Hann window of `window-length`." [window-length] (call (ns-fn tf/signal "hannWindow") window-length))
(defn signal-frame
  "Frames a 1-D signal into [frames frame-length] windows stepping by
   `frame-step`; opts {:pad-end false :pad-value 0}."
  ([signal frame-length frame-step] (signal-frame signal frame-length frame-step nil))
  ([signal frame-length frame-step {:keys [pad-end pad-value]}] (call (ns-fn tf/signal "frame") signal frame-length frame-step pad-end pad-value)))
(defn signal-stft
  "Short-time Fourier transform (complex64) of a 1-D signal; opts
   {:fft-length n :window-fn (fn [length] -> tensor)} (default Hann)."
  ([signal frame-length frame-step] (signal-stft signal frame-length frame-step nil))
  ([signal frame-length frame-step {:keys [fft-length window-fn]}] (call (ns-fn tf/signal "stft") signal frame-length frame-step fft-length window-fn)))

;; Misc (3)
(defn moving-average
  "Exponential moving average: v' = decay·v + (1-decay)·x; opts {:step n
   :zero-debias true}."
  ([v x decay] (moving-average v x decay nil))
  ([v x decay {:keys [step zero-debias]}] (call tf/movingAverage v x decay step zero-debias)))

;; image.* (16)
(defn- image-fn [name] (ns-fn tf/image name))
(def image-flip-left-right "Flips [batch h w c] images horizontally." (unary (image-fn "flipLeftRight")))
(def image-grayscale-to-rgb "Repeats the single channel of [.. h w 1] images to 3." (unary (image-fn "grayscaleToRGB")))
(def image-rgb-to-grayscale "Luma-weighted [.. h w 3] -> [.. h w 1]." (unary (image-fn "rgbToGrayscale")))
(defn image-resize-bilinear
  "Bilinear resize of [batch h w c] or [h w c] images to `size` [h w]; opts
   {:align-corners false :half-pixel-centers false}."
  ([images size] (image-resize-bilinear images size nil))
  ([images size {:keys [align-corners half-pixel-centers]}] (call (image-fn "resizeBilinear") images size align-corners half-pixel-centers)))
(defn image-resize-nearest-neighbor
  "Nearest-neighbour resize to `size` [h w]; opts as image-resize-bilinear."
  ([images size] (image-resize-nearest-neighbor images size nil))
  ([images size {:keys [align-corners half-pixel-centers]}] (call (image-fn "resizeNearestNeighbor") images size align-corners half-pixel-centers)))
(defn image-rotate-with-offset
  "Rotates [batch h w c] images by `radians`; opts {:fill-value 0|[r g b]
   :center 0.5|[x y]}."
  ([image radians] (image-rotate-with-offset image radians nil))
  ([image radians {:keys [fill-value center]}] (call (image-fn "rotateWithOffset") image radians fill-value center)))
(defn image-crop-and-resize
  "Crops `boxes` ([n 4] normalised y1 x1 y2 x2) of the images indexed by
   `box-ind` and resizes each to `crop-size` [h w]; opts {:method :bilinear|:nearest
   :extrapolation-value 0}."
  ([image boxes box-ind crop-size] (image-crop-and-resize image boxes box-ind crop-size nil))
  ([image boxes box-ind crop-size {:keys [method extrapolation-value]}]
   (call (image-fn "cropAndResize") image boxes box-ind crop-size (enum-arg method) extrapolation-value)))
(defn image-non-max-suppression
  "int32 indices of the boxes ([n 4]) kept by NMS on `scores`; opts
   {:iou-threshold 0.5 :score-threshold -Infinity}."
  ([boxes scores max-output-size] (image-non-max-suppression boxes scores max-output-size nil))
  ([boxes scores max-output-size {:keys [iou-threshold score-threshold]}]
   (call (image-fn "nonMaxSuppression") boxes scores max-output-size iou-threshold score-threshold)))
(defn image-non-max-suppression!
  "Promise form of `image-non-max-suppression`."
  ([boxes scores max-output-size] (image-non-max-suppression! boxes scores max-output-size nil))
  ([boxes scores max-output-size {:keys [iou-threshold score-threshold]}]
   (async-guard #((image-fn "nonMaxSuppressionAsync") (->arg boxes) (->arg scores) max-output-size (undef-if-nil iou-threshold) (undef-if-nil score-threshold)))))
(defn image-non-max-suppression-with-score
  "{:selected-indices t :selected-scores t} soft-NMS; opts {:iou-threshold
   :score-threshold :soft-nms-sigma}."
  ([boxes scores max-output-size] (image-non-max-suppression-with-score boxes scores max-output-size nil))
  ([boxes scores max-output-size {:keys [iou-threshold score-threshold soft-nms-sigma]}]
   (call (image-fn "nonMaxSuppressionWithScore") boxes scores max-output-size iou-threshold score-threshold soft-nms-sigma)))
(defn image-non-max-suppression-with-score!
  "Promise form of `image-non-max-suppression-with-score`."
  ([boxes scores max-output-size] (image-non-max-suppression-with-score! boxes scores max-output-size nil))
  ([boxes scores max-output-size {:keys [iou-threshold score-threshold soft-nms-sigma]}]
   (-> (async-guard #((image-fn "nonMaxSuppressionWithScoreAsync") (->arg boxes) (->arg scores) max-output-size (undef-if-nil iou-threshold) (undef-if-nil score-threshold) (undef-if-nil soft-nms-sigma)))
       (p/then ->result))))
(defn image-non-max-suppression-padded
  "{:selected-indices t :valid-outputs t} NMS padded to `max-output-size`;
   opts {:iou-threshold :score-threshold :pad-to-max-output-size false}."
  ([boxes scores max-output-size] (image-non-max-suppression-padded boxes scores max-output-size nil))
  ([boxes scores max-output-size {:keys [iou-threshold score-threshold pad-to-max-output-size]}]
   (call (image-fn "nonMaxSuppressionPadded") boxes scores max-output-size iou-threshold score-threshold pad-to-max-output-size)))
(defn image-non-max-suppression-padded!
  "Promise form of `image-non-max-suppression-padded`."
  ([boxes scores max-output-size] (image-non-max-suppression-padded! boxes scores max-output-size nil))
  ([boxes scores max-output-size {:keys [iou-threshold score-threshold pad-to-max-output-size]}]
   (-> (async-guard #((image-fn "nonMaxSuppressionPaddedAsync") (->arg boxes) (->arg scores) max-output-size (undef-if-nil iou-threshold) (undef-if-nil score-threshold) (undef-if-nil pad-to-max-output-size)))
       (p/then ->result))))
(defn image-threshold
  "Binarises a [h w c] image; opts {:method :binary|:otsu :inverted false
   :thresh-value 0.5}."
  ([image] (image-threshold image nil))
  ([image {:keys [method inverted thresh-value]}] (call (image-fn "threshold") image (enum-arg method) inverted thresh-value)))
(defn image-transform
  "Projective transform of [batch h w c] images by `transforms` ([1 8] or
   [batch 8]); opts {:interpolation :nearest|:bilinear :fill-mode
   :constant|:reflect|:wrap|:nearest :fill-value 0 :output-shape [h w]}."
  ([image transforms] (image-transform image transforms nil))
  ([image transforms {:keys [interpolation fill-mode fill-value output-shape]}]
   (call (image-fn "transform") image transforms (enum-arg interpolation) (enum-arg fill-mode) fill-value output-shape)))

;; losses.* (9) as functions for custom loops
(def loss-reductions
  "Loss reduction keywords -> tfjs Reduction enum values."
  {:none 0 :mean 1 :sum 2 :sum-by-nonzero-weights 3})
(defn- reduction-arg [r]
  (cond (nil? r) js/undefined
        (contains? loss-reductions r) (get loss-reductions r)
        (number? r) r
        :else (throw (util/argument-error (str "unknown reduction " (pr-str r)) {:allowed (set (keys loss-reductions))}))))
(defn- loss-fn [name] (ns-fn tf/losses name))
(defn- simple-loss [name]
  (fn ([labels predictions] ((loss-fn name) (->arg labels) (->arg predictions)))
    ([labels predictions {:keys [weights reduction]}]
     (call (loss-fn name) labels predictions weights (reduction-arg reduction)))))
(def losses-absolute-difference "Mean |labels - predictions|; opts {:weights t :reduction :mean}." (simple-loss "absoluteDifference"))
(def losses-mean-squared-error "Mean (labels - predictions)²; opts {:weights t :reduction :mean}." (simple-loss "meanSquaredError"))
(def losses-hinge-loss "Hinge loss; opts {:weights t :reduction :mean}." (simple-loss "hingeLoss"))
(defn losses-compute-weighted-loss "Reduces elementwise `losses` with optional weights; opts {:weights :reduction}." ([losses] (call (loss-fn "computeWeightedLoss") losses)) ([losses {:keys [weights reduction]}] (call (loss-fn "computeWeightedLoss") losses weights (reduction-arg reduction))))
(defn losses-cosine-distance "1 - cosine similarity along `axis`; opts {:weights :reduction}." ([labels predictions axis] (call (loss-fn "cosineDistance") labels predictions axis)) ([labels predictions axis {:keys [weights reduction]}] (call (loss-fn "cosineDistance") labels predictions axis weights (reduction-arg reduction))))
(defn losses-huber-loss "Huber loss; opts {:weights :delta 1 :reduction}." ([labels predictions] (call (loss-fn "huberLoss") labels predictions)) ([labels predictions {:keys [weights delta reduction]}] (call (loss-fn "huberLoss") labels predictions weights delta (reduction-arg reduction))))
(defn losses-log-loss "Log loss; opts {:weights :epsilon 1e-7 :reduction}." ([labels predictions] (call (loss-fn "logLoss") labels predictions)) ([labels predictions {:keys [weights epsilon reduction]}] (call (loss-fn "logLoss") labels predictions weights epsilon (reduction-arg reduction))))
(defn losses-sigmoid-cross-entropy "Sigmoid cross-entropy of multi-class labels vs logits; opts {:weights :label-smoothing 0 :reduction}." ([labels logits] (call (loss-fn "sigmoidCrossEntropy") labels logits)) ([labels logits {:keys [weights label-smoothing reduction]}] (call (loss-fn "sigmoidCrossEntropy") labels logits weights label-smoothing (reduction-arg reduction))))
(defn losses-softmax-cross-entropy "Softmax cross-entropy of one-hot labels vs logits; opts {:weights :label-smoothing 0 :reduction}." ([labels logits] (call (loss-fn "softmaxCrossEntropy") labels logits)) ([labels logits {:keys [weights label-smoothing reduction]}] (call (loss-fn "softmaxCrossEntropy") labels logits weights label-smoothing (reduction-arg reduction))))

;; string.* (4)
(defn string-n-grams
  "{:n-grams t :n-grams-splits t} n-grams of a ragged string tensor
   (`data` 1-D strings, `data-splits` int32 row splits): (string-n-grams
   data splits separator n-gram-widths left-pad right-pad pad-width
   preserve-short-sequences)."
  [data data-splits separator n-gram-widths left-pad right-pad pad-width preserve-short-sequences]
  (call (ns-fn tf/string "stringNGrams") data data-splits separator n-gram-widths left-pad right-pad pad-width preserve-short-sequences))
(defn string-split
  "{:indices t :values t :shape t} sparse split of a 1-D string tensor on
   `delimiter` (a string); `skip-empty` (default true)."
  ([input delimiter] (call (ns-fn tf/string "stringSplit") input delimiter))
  ([input delimiter skip-empty] (call (ns-fn tf/string "stringSplit") input delimiter skip-empty)))
(defn string-to-hash-bucket-fast "int32 bucket ids (0 .. num-buckets-1) of a string tensor." [input num-buckets] (call (ns-fn tf/string "stringToHashBucketFast") input num-buckets))
(defn static-regex-replace "Replaces `pattern` (a regex string) with `rewrite` in a string tensor; `replace-global` (default true)." ([input pattern rewrite] (call (ns-fn tf/string "staticRegexReplace") input pattern rewrite)) ([input pattern rewrite replace-global] (call (ns-fn tf/string "staticRegexReplace") input pattern rewrite replace-global)))

;; ---------------------------------------------------------------------------
;; Training utilities (§6.2): optimizers, gradients, minimize!
;; ---------------------------------------------------------------------------

(defn make-optimizer
  "Opaque optimizer from a tf.train constructor name (\"sgd\" \"momentum\"
   \"rmsprop\" \"adam\" \"adadelta\" \"adamax\" \"adagrad\") and its positional
   arguments (nil -> tfjs default). tensorflow-cljs.models/optimizer offers
   the kebab option-map form."
  [ctor-name args]
  (let [ctor (ns-fn tf/train ctor-name)]
    (when-not ctor
      (throw (util/argument-error (str "unknown optimizer " (pr-str ctor-name)) {:optimizer ctor-name})))
    (guard #(apply ctor (map undef-if-nil args)))))

(defn optimizer-class-name "\"SGD\", \"Adam\" .. of an opaque optimizer." [o] (js-invoke o "getClassName"))

(defn optimizer-dispose!
  "Frees an optimizer's internal tensors; returns nil. Needed because SGD
   keeps its learning rate and Momentum its momentum as plain (kept) scalars
   that `dispose-variables!` does not see (probe-verified: sgd 1, momentum 2;
   adam / adamax / rmsprop / adagrad / adadelta use variables). tfjs 4.22's
   Momentum.dispose frees only one of its two scalars."
  [o]
  (guard #(js-invoke o "dispose")) nil)

(defn minimize!
  "One optimizer step: `f` is a zero-argument fn returning a scalar loss
   tensor; every trainable variable it touched (or `var-list`) is updated.
   Returns the loss as a number when `return-cost?` (default true), else nil.
   Synchronous."
  ([optimizer f] (minimize! optimizer f nil))
  ([optimizer f {:keys [return-cost? var-list] :or {return-cost? true}}]
   (guard #(let [cost (js-invoke optimizer "minimize" f return-cost? (if var-list (to-array var-list) js/undefined))]
             (when (tensor? cost)
               (let [v (aget (js-invoke cost "dataSync") 0)] (js-invoke cost "dispose") v))))))

(defn compute-gradients
  "{:value scalar-tensor :grads {\"var-name\" tensor ..}} of `f` (zero-arg fn
   returning a scalar tensor) w.r.t. the trainable variables (or `var-list`)."
  ([optimizer f] (compute-gradients optimizer f nil))
  ([optimizer f var-list]
   (guard #(let [r (js-invoke optimizer "computeGradients" f (if var-list (to-array var-list) js/undefined))]
             {:value (gobj/get r "value") :grads (element->clj (gobj/get r "grads"))}))))

(defn apply-gradients!
  "Applies a map of variable-name -> gradient tensor with `optimizer`."
  [optimizer grads]
  (guard #(js-invoke optimizer "applyGradients" (->js-container grads))) nil)

(defn grad
  "Gradient function of a scalar fn of one tensor: ((grad f) x) -> dy/dx
   tensor; optional second argument `dy` seeds the upstream gradient."
  [f]
  (let [g (tf/grad f)]
    (fn ([x] (call g x)) ([x dy] (call g x dy)))))

(defn grads
  "Gradient function of a scalar fn of several tensors: ((grads f) [x y])
   -> vector of gradients."
  [f]
  (let [g (tf/grads (fn [& xs] (f (vec xs))))]
    (fn ([xs] (call g xs)) ([xs dy] (call g xs dy)))))

(defn value-and-grad
  "((value-and-grad f) x) -> {:value t :grad t}."
  [f]
  (let [g (tf/valueAndGrad f)]
    (fn [x] (guard #(let [r (g (->arg x))] {:value (gobj/get r "value") :grad (gobj/get r "grad")})))))

(defn variable-grads
  "{:value scalar :grads {\"name\" tensor ..}}: gradients of the scalar
   returned by zero-arg fn `f` w.r.t. all trainable variables (or `var-list`)."
  ([f] (variable-grads f nil))
  ([f var-list]
   (guard #(let [r (tf/variableGrads f (if var-list (to-array var-list) js/undefined))]
             {:value (gobj/get r "value") :grads (element->clj (gobj/get r "grads"))}))))

(defn custom-grad
  "Op with a custom gradient: `f` receives the input tensors (as a vector)
   and must return {:value tensor :grad-func (fn [dy] -> vector of grads)}.
   Returns a fn of tensors."
  [f]
  (let [g (tf/customGrad (fn [& args]
                           (let [xs (vec (butlast args))
                                 {:keys [value grad-func]} (f xs)]
                             #js {:value value
                                  :gradFunc (fn [dy _saved] (->js-container (grad-func dy)))})))]
    (fn [& xs] (apply call g xs))))

;; ---------------------------------------------------------------------------
;; Layer bridge (consumed by tensorflow-cljs.layers)
;; ---------------------------------------------------------------------------

(defn make-layer
  "Opaque layer from a tf.layers constructor name (\"dense\", \"conv2d\" ..)
   and a kebab config map (translated with util/->config: :kernel-initializer
   :glorot-uniform -> kernelInitializer \"glorotUniform\"; opaque initializer /
   regularizer / constraint / layer values pass through)."
  [ctor-name config]
  (let [ctor (ns-fn tf/layers ctor-name)]
    (when-not ctor
      (throw (util/argument-error (str "unknown layer constructor " (pr-str ctor-name)) {:layer ctor-name})))
    (guard #(ctor (->config-js (or config {}))))))

(defn make-input
  "Symbolic input tensor from a kebab config {:shape [..] :batch-shape [..]
   :name .. :dtype .. :sparse ..} (nil = unknown batch dim)."
  [config]
  (guard #(tf/input (->config-js config))))

(defn make-initializer
  "Opaque initializer from a tf.initializers name and kebab args."
  [name args]
  (let [ctor (ns-fn tf/initializers name)]
    (when-not ctor (throw (util/argument-error (str "unknown initializer " (pr-str name)) {:initializer name})))
    (guard #(ctor (->config-js (or args {}))))))

(defn make-regularizer
  "Opaque regularizer from a tf.regularizers name (\"l1\" \"l2\" \"l1l2\") and args."
  [name args]
  (let [ctor (ns-fn tf/regularizers name)]
    (when-not ctor (throw (util/argument-error (str "unknown regularizer " (pr-str name)) {:regularizer name})))
    (guard #(ctor (->config-js (or args {}))))))

(defn make-constraint
  "Opaque constraint from a tf.constraints name and args."
  [name args]
  (let [ctor (ns-fn tf/constraints name)]
    (when-not ctor (throw (util/argument-error (str "unknown constraint " (pr-str name)) {:constraint name})))
    (guard #(ctor (->config-js (or args {}))))))

(defn layer-apply
  "Applies a layer: symbolic input(s) -> symbolic output (graph building),
   concrete tensor(s) -> tensor (eager). `inputs` is one value or a vector;
   `kwargs` an optional kebab map (e.g. {:training true})."
  ([layer inputs] (layer-apply layer inputs nil))
  ([layer inputs kwargs]
   (guard #(->result (js-invoke layer "apply" (->js-container inputs) (if kwargs (->config-js kwargs) js/undefined))))))

(defn layer-weights
  "Vector of the layer's weight tensors (kernel, bias, ..); `trainable-only?`."
  ([layer] (layer-weights layer false))
  ([layer trainable-only?] (guard #(->result (js-invoke layer "getWeights" trainable-only?)))))

(defn layer-set-weights!
  "Sets the layer's weights from a vector of tensors (same shapes, same order
   as `layer-weights`); returns the layer."
  [layer weights]
  (guard #(js-invoke layer "setWeights" (to-array weights))) layer)

(defn layer-count-params "Number of scalar parameters of a built layer." [layer] (guard #(js-invoke layer "countParams")))
(defn layer-config "Kebab map of the layer's configuration (getConfig)." [layer] (guard #(js->kebab (js-invoke layer "getConfig"))))
(defn layer-name "The layer's unique name (\"dense_Dense1\")." [layer] (gobj/get layer "name"))
(defn layer-class-name "The layer's Keras class name (\"Dense\")." [layer] (js-invoke layer "getClassName"))
(defn layer-trainable? "True when the layer's weights are updated by training." [layer] (gobj/get layer "trainable"))
(defn layer-set-trainable! "Freezes (false) or unfreezes the layer; returns it." [layer trainable?] (gobj/set layer "trainable" trainable?) layer)
(defn layer-built? "True once the layer has weights (after apply / build)." [layer] (gobj/get layer "built"))
(defn layer-build! "Builds the layer's weights for `input-shape` (nil = unknown batch dim); returns it." [layer input-shape] (guard #(js-invoke layer "build" (->js-container input-shape))) layer)
(defn layer-output-shape "Output shape (nil = batch dim) of a built layer; a vector of shapes for multi-output layers." [layer] (guard #(js-array->clj (gobj/get layer "outputShape"))))
(defn layer-compute-output-shape "Output shape for a given `input-shape` (nil = batch dim)." [layer input-shape] (guard #(js-array->clj (js-invoke layer "computeOutputShape" (->js-container input-shape)))))
(defn layer-dispose! "Frees the layer's weights; returns {:ref-count-after-dispose n :num-disposed-variables n}." [layer] (guard #(js->kebab (js-invoke layer "dispose"))))
(defn layer-reset-states! "Resets a stateful RNN layer's states; returns the layer." [layer] (guard #(js-invoke layer "resetStates")) layer)
(defn layer-batch-input-shape "The batch input shape of an input-carrying layer, or nil." [layer] (js-array->clj (gobj/get layer "batchInputShape")))
(defn layer-input "Symbolic input(s) of a layer used in a graph." [layer] (guard #(->result (gobj/get layer "input"))))
(defn layer-output "Symbolic output(s) of a layer used in a graph." [layer] (guard #(->result (gobj/get layer "output"))))

(defn symbolic-name "Name of a symbolic tensor." [s] (gobj/get s "name"))
(defn symbolic-shape "Shape of a symbolic tensor (nil = unknown batch dim)." [s] (js-array->clj (gobj/get s "shape")))
(defn symbolic-dtype "Dtype keyword of a symbolic tensor." [s] (util/str->dtype (gobj/get s "dtype")))

(defn metric-fn
  "tf.metrics function by name (\"binaryAccuracy\" \"mse\" \"r2Score\" ..) as a
   ClojureScript fn of (y-true y-pred) -> tensor, for custom loops."
  [name]
  (let [f (ns-fn tf/metrics name)]
    (when-not f (throw (util/argument-error (str "unknown metric function " (pr-str name)) {:metric name})))
    (fn [y-true y-pred] (call f y-true y-pred))))

;; ---------------------------------------------------------------------------
;; Model bridge (consumed by tensorflow-cljs.models)
;; ---------------------------------------------------------------------------

(defn make-sequential
  "Opaque sequential model from a vector of layers (the first must carry
   :input-shape) and an optional name."
  [layers name]
  (guard #(tf/sequential (->config-js {:layers (vec layers) :name name}))))

(defn make-model
  "Opaque functional model from symbolic :inputs / :outputs (each one value or
   a vector) and an optional :name."
  [{:keys [inputs outputs name]}]
  (guard #(tf/model (->config-js {:inputs inputs :outputs outputs :name name}))))

(defn model-add! "Appends a layer to a sequential model; returns the model." [model layer] (guard #(js-invoke model "add" layer)) model)
(defn model-pop! "Removes the last layer of a sequential model; returns the model." [model] (guard #(js-invoke model "pop")) model)

(defn model-compile!
  "Compiles with an already-translated config {:optimizer string|opaque
   :loss string|fn|vector|map :metrics vector|map}; returns the model."
  [model config]
  (guard #(js-invoke model "compile" (->config-js config))) model)

(defn- logs->clj [logs] (if (nil? logs) {} (js->kebab logs)))

(def callback-hooks
  "Callback map keys -> tfjs hook names and arities."
  {:on-train-begin ["onTrainBegin" 1] :on-train-end ["onTrainEnd" 1]
   :on-epoch-begin ["onEpochBegin" 2] :on-epoch-end ["onEpochEnd" 2]
   :on-batch-begin ["onBatchBegin" 2] :on-batch-end ["onBatchEnd" 2]
   :on-yield ["onYield" 3]})

(defn make-callback
  "Opaque callback from a map of hook fns {:on-epoch-end (fn [epoch logs])
   :on-batch-end (fn [batch logs]) :on-train-begin (fn [logs]) :on-train-end
   (fn [logs]) :on-epoch-begin .. :on-batch-begin .. :on-yield (fn [epoch
   batch logs])}; `logs` arrives as a kebab map of numbers; a hook may return
   a promise (awaited). `yield-every` (:auto :batch :epoch :never or ms)
   decides when :on-yield fires — it is a constructor argument of tfjs's
   CustomCallback, NOT read from the fit options, so fit! passes its own
   :yield-every here (default :auto = every 125 ms of wall time, which a
   short fit never reaches)."
  ([hooks] (make-callback hooks nil))
  ([hooks yield-every]
   (let [o (js-obj)]
     (doseq [[k f] hooks]
       (let [[js-name arity] (get callback-hooks k)]
         (when-not js-name
           (throw (util/argument-error (str "unknown callback hook " (pr-str k)) {:hook k :allowed (set (keys callback-hooks))})))
         (gobj/set o js-name
                   (case arity
                     1 (fn [logs] (f (logs->clj logs)))
                     2 (fn [i logs] (f i (logs->clj logs)))
                     3 (fn [e b logs] (f e b (logs->clj logs)))))))
     (new tf/CustomCallback o (enum-arg yield-every)))))

(defn early-stopping
  "Opaque EarlyStopping callback; `config` already translated ({:monitor
   \"val_loss\" :min-delta 0 :patience 0 :mode \"auto\" :baseline n
   :restore-best-weights false :verbose 0})."
  [config]
  (guard #((ns-fn tf/callbacks "earlyStopping") (->config-js config))))

(defn- callbacks-arg
  "A map of hooks, a vector of maps / opaque callbacks, or nil -> JS array of
   BaseCallback instances (tfjs rejects mixed plain-object + instance lists).
   Hook maps get the fit's `yield-every` so :on-yield fires as configured."
  [cbs yield-every]
  (cond
    (nil? cbs) js/undefined
    (map? cbs) (array (make-callback cbs yield-every))
    (sequential? cbs) (to-array (map (fn [c] (cond (map? c) (make-callback c yield-every)
                                                    (base-callback? c) c
                                                    :else (throw (util/argument-error "callbacks must be hook maps or callback objects" {:callback c}))))
                                     cbs))
    (base-callback? cbs) (array cbs)
    :else (throw (util/argument-error "callbacks must be a hook map or a vector" {:callbacks cbs}))))

(defn- fit-args-js
  "Fit / fit-dataset option map -> JS args (callbacks bridged, tensors kept)."
  [opts]
  (let [o (->config-js (dissoc opts :callbacks :validation-data :class-weight))]
    (when (contains? opts :callbacks) (gobj/set o "callbacks" (callbacks-arg (:callbacks opts) (:yield-every opts))))
    (when-let [vd (:validation-data opts)] (gobj/set o "validationData" (->js-container vd)))
    (when-let [cw (:class-weight opts)] (gobj/set o "classWeight" (->js-container cw)))
    o))

(defn- history->clj
  "tfjs History -> {:epoch [..] :history {:loss [..] :val-loss [..]} :params {..}}."
  [h]
  {:epoch (js-array->clj (gobj/get h "epoch"))
   :history (js->kebab (gobj/get h "history"))
   :params (js->kebab (gobj/get h "params"))})

(defn model-fit!
  "Promise of the kebab history of training `model` on tensors (or vectors /
   maps of tensors) `x` / `y` with fit options (already kebab; translated
   here)."
  [model x y opts]
  (-> (async-guard #(js-invoke model "fit" (->js-container x) (->js-container y) (fit-args-js opts)))
      (p/then history->clj)))

(defn model-fit-dataset!
  "Promise of the kebab history of training on an opaque dataset of {:xs ..
   :ys ..} batches."
  [model dataset opts]
  (-> (async-guard #(js-invoke model "fitDataset" dataset (fit-args-js opts)))
      (p/then history->clj)))

(defn model-train-on-batch!
  "Promise of the loss (number) or [loss metric ..] vector for one gradient
   step on a batch."
  [model x y]
  (-> (async-guard #(js-invoke model "trainOnBatch" (->js-container x) (->js-container y)))
      (p/then (fn [r] (if (array? r) (vec r) r)))))

(defn model-predict
  "Opaque output tensor (or vector for multi-output models) for `x` (a
   tensor, or a vector of tensors for multi-input models — tfjs predict does
   not take the named map that fit accepts); opts {:batch-size 32 :verbose
   false}."
  [model x opts]
  (guard #(->result (js-invoke model "predict" (->js-container x) (->config-js (or opts {}))))))

(defn model-predict-on-batch
  "Opaque output for a single batch `x` (no batching)."
  [model x]
  (guard #(->result (js-invoke model "predictOnBatch" (->js-container x)))))

(defn model-metrics-names
  "Vector of metric names after compile ([\"loss\" \"acc\"]); nil before."
  [model]
  (some-> (gobj/get model "metricsNames") js-array->clj))

(defn- scalars->metrics
  "Scalar or array of scalars -> {:loss n :acc n} keyed by metric names; the
   scalars are disposed."
  [model r]
  (let [scalars (if (array? r) (vec (array-seq r)) [r])
        names (or (model-metrics-names model) (mapv (fn [i] (str "metric-" i)) (cljs.core/range (count scalars))))
        vals (mapv (fn [s] (let [v (aget (js-invoke s "dataSync") 0)] (js-invoke s "dispose") v)) scalars)]
    (zipmap (map util/camel->kebab names) vals)))

(defn model-evaluate
  "Kebab map metric-name -> number of evaluating tensors `x` / `y`; opts
   {:batch-size :verbose :sample-weight :steps}."
  [model x y opts]
  (guard #(scalars->metrics model (js-invoke model "evaluate" (->js-container x) (->js-container y) (->config-js (or opts {}))))))

(defn model-evaluate-dataset!
  "Promise of the kebab metrics map of evaluating an opaque dataset; opts
   {:batches n :verbose}."
  [model dataset opts]
  (-> (async-guard #(js-invoke model "evaluateDataset" dataset (->config-js (or opts {}))))
      (p/then (fn [r] (scalars->metrics model r)))))

(defn model-summary
  "Vector of the summary lines tfjs would print; opts {:line-length n
   :positions [..]}."
  [model {:keys [line-length positions]}]
  (let [lines (volatile! [])]
    (guard #(js-invoke model "summary" (undef-if-nil line-length) (if positions (to-array positions) js/undefined)
                      (fn [& args] (vswap! lines conj (str/join " " args)))))
    @lines))

(defn model-layers "Vector of the model's opaque layers in order." [model] (vec (array-seq (gobj/get model "layers"))))
(defn model-get-layer
  "Layer by name (string) or index (number); throws :value-error when absent."
  [model name-or-index]
  (guard #(if (number? name-or-index)
            (js-invoke model "getLayer" js/undefined name-or-index)
            (js-invoke model "getLayer" name-or-index))))
(defn model-name "The model's name." [model] (gobj/get model "name"))
(defn model-weights "Vector of every weight tensor (layer order: kernel, bias ..); `trainable-only?`." ([model] (model-weights model false)) ([model trainable-only?] (guard #(->result (js-invoke model "getWeights" trainable-only?)))))
(defn model-set-weights! "Sets every weight from a vector of tensors; returns the model." [model weights] (guard #(js-invoke model "setWeights" (to-array weights))) model)
(defn model-count-params "Total number of parameters." [model] (guard #(js-invoke model "countParams")))
(defn model-compiled? "True once compiled (an optimizer is attached)." [model] (some? (gobj/get model "optimizer")))
(defn model-optimizer "The opaque optimizer attached by compile, or nil." [model] (gobj/get model "optimizer"))
(defn model-stop-training! "Asks a running fit to stop after the current epoch (call from a callback)." [model] (gobj/set model "stopTraining" true) nil)
(defn model-stop-training? "True when a stop was requested." [model] (boolean (gobj/get model "stopTraining")))
(defn model-reset-stop-training! "Clears the stop-training flag (fit clears it itself before starting)." [model] (gobj/set model "stopTraining" false) nil)
(defn model-dispose!
  "Frees the model's weights and, unless `keep-optimizer?`, the attached
   optimizer's tensors (tfjs's model.dispose leaves the optimizer alone;
   see `optimizer-dispose!`). Returns {:ref-count-after-dispose n
   :num-disposed-variables n}."
  ([model] (model-dispose! model false))
  ([model keep-optimizer?]
   (let [r (guard #(js->kebab (js-invoke model "dispose")))]
     (when-not keep-optimizer?
       (when-let [o (gobj/get model "optimizer")] (optimizer-dispose! o)))
     r)))
(defn model-inputs "Vector of the model's symbolic inputs." [model] (vec (array-seq (gobj/get model "inputs"))))
(defn model-outputs "Vector of the model's symbolic outputs." [model] (vec (array-seq (gobj/get model "outputs"))))
(defn model-input-names "Vector of input layer names." [model] (js-array->clj (gobj/get model "inputNames")))
(defn model-output-names "Vector of output layer names." [model] (js-array->clj (gobj/get model "outputNames")))
(defn model-to-json
  "The Keras JSON topology as ClojureScript data with string keys verbatim
   ({\"class_name\" .. \"config\" ..})."
  [model]
  (guard #(js->clj (js-invoke model "toJSON" nil false))))
(defn model-from-json!
  "Promise of a model rebuilt from a topology (as returned by model-to-json)
   — weights are freshly initialised."
  [topology]
  (async-guard #((ns-fn tf/models "modelFromJSON") #js {:modelTopology (clj->js topology)})))
(defn model-user-metadata "The user-defined metadata map saved with the model, or nil." [model] (some-> (js-invoke model "getUserDefinedMetadata") js->clj))
(defn model-set-user-metadata! "Attaches a JSON-able ClojureScript map (string keys kept verbatim); returns the model." [model m] (guard #(js-invoke model "setUserDefinedMetadata" (clj->js m))) model)
(defn model-execute
  "Vector of intermediate output tensors for `x` at `output-names`: layer
   names (resolved to the layer's output symbolic tensor) or symbolic tensor
   names (strings)."
  [model x output-names]
  (let [by-layer (into {} (map (fn [l] [(layer-name l) l])) (array-seq (gobj/get model "layers")))
        names (mapv (fn [n] (if-let [l (get by-layer n)] (symbolic-name (js-invoke l "getOutputAt" 0)) n)) output-names)]
    (guard #(->result (js-invoke model "execute" (->js-container x) (to-array names))))))

;; ---------------------------------------------------------------------------
;; IO bridge: in-memory artifacts + the library's own file handler (§4.6)
;; ---------------------------------------------------------------------------

(defn- save-config-js [opts] (->config-js (select-keys (or opts {}) [:trainable-only :include-optimizer])))

(defn model-artifacts!
  "Promise of the opaque ModelArtifacts of `model` (in memory); opts
   {:include-optimizer false :trainable-only false}. :include-optimizer
   stores the training config so a loaded model is already compiled."
  [model opts]
  (let [captured (volatile! nil)
        handler ((gobj/get tf/io "withSaveHandler")
                 (fn [artifacts]
                   (vreset! captured artifacts)
                   (js/Promise.resolve #js {:modelArtifactsInfo ((gobj/get tf/io "getModelArtifactsInfoForJSON") artifacts)})))]
    (-> (async-guard #(js-invoke model "save" handler (save-config-js opts)))
        (p/then (fn [_] @captured)))))

(defn artifacts-info
  "{:date-saved .. :model-topology-type \"JSON\" :model-topology-bytes n
   :weight-specs-bytes n :weight-data-bytes n} of opaque artifacts."
  [artifacts]
  (js->kebab ((gobj/get tf/io "getModelArtifactsInfoForJSON") artifacts)))

(defn artifacts-weight-specs
  "Vector of {:name \"dense_Dense1/kernel\" :shape [..] :dtype \"float32\"} of
   opaque artifacts."
  [artifacts]
  (mapv js->kebab (array-seq (gobj/get artifacts "weightSpecs"))))

(defn artifacts-format
  "\"layers-model\" (Keras / tfjs-layers) or \"graph-model\" (tfjs-converter
   output) of opaque artifacts; nil when the model.json carries no format."
  [artifacts]
  (gobj/get artifacts "format"))

(defn load-artifacts!
  "Promise of a layers model loaded from opaque artifacts (in memory).
   Rejects with :tensorflow-cljs/io-error when the artifacts are a graph
   model (tfjs would fail with an opaque \"Improper config format\"
   ValueError): use `load-graph-artifacts!`."
  [artifacts]
  (if (= "graph-model" (artifacts-format artifacts))
    (p/rejected (util/io-error "artifacts are a graph model (format \"graph-model\"): use load-graph! / load-graph-artifacts!"
                               {:format "graph-model"}))
    (async-guard #(tf/loadLayersModel ((gobj/get tf/io "fromMemory") artifacts)))))

(defn- weight-data->array-buffer
  "ModelArtifacts.weightData (ArrayBuffer or ArrayBuffer[]) -> one ArrayBuffer."
  [wd]
  (if (array? wd) ((gobj/get tf/io "concatenateArrayBuffers") wd) wd))

(defn- node-buffer->array-buffer [b]
  (js-invoke (gobj/get b "buffer") "slice" (gobj/get b "byteOffset") (+ (gobj/get b "byteOffset") (gobj/get b "byteLength"))))

(defn- strip-file-scheme [s] (str/replace s #"^file://" ""))

(defn- model-dir-and-json
  "\"dir\", \"dir/model.json\" or \"file://..\" -> [dir json-path]."
  [p]
  (let [p (strip-file-scheme p)]
    (if (str/ends-with? p ".json")
      [(path/dirname p) p]
      [p (path/join p "model.json")])))

(defn save-artifacts-to-dir!
  "Writes opaque artifacts as `model.json` + one `weights.bin` shard into
   directory `dir` (created if needed) — the tfjs-node / converter layout.
   Returns a promise of {:dir .. :model-json .. :weights-bin ..
   :model-artifacts-info {..}}."
  [artifacts dir]
  (let [[dir json-path] (model-dir-and-json dir)
        weights-path (path/join dir "weights.bin")]
    (p/do
      (try
        (fs/mkdirSync dir #js {:recursive true})
        (let [specs (gobj/get artifacts "weightSpecs")
              manifest #js [#js {:paths #js ["./weights.bin"] :weights specs}]
              model-json (js-obj)]
          (doseq [k ["modelTopology" "format" "generatedBy" "convertedBy" "trainingConfig"
                     "userDefinedMetadata" "signature" "modelInitializer" "initializerSignature"]]
            (when-some [v (gobj/get artifacts k)] (gobj/set model-json k v)))
          (gobj/set model-json "weightsManifest" manifest)
          (fs/writeFileSync json-path (js/JSON.stringify model-json))
          (fs/writeFileSync weights-path (js/Buffer.from (weight-data->array-buffer (gobj/get artifacts "weightData"))))
          {:dir dir :model-json json-path :weights-bin weights-path
           :model-artifacts-info (artifacts-info artifacts)})
        (catch :default e
          (throw (if (some? (ex-data e)) e
                     (util/io-error (str "cannot write model to " dir ": " (gobj/get e "message")) {:dir dir :cause e}))))))))

(defn read-artifacts-from-dir!
  "Promise of opaque artifacts read from `model.json` (+ the weight shards of
   its manifest, concatenated in manifest order) in directory `dir` (or a
   model.json path, or a \"file://\" URL). Rejects with
   :tensorflow-cljs/io-error when files are missing or malformed."
  [dir]
  (let [[dir json-path] (model-dir-and-json dir)]
    (p/do
      (try
        (when-not (fs/existsSync json-path)
          (throw (util/io-error (str "model.json not found: " json-path) {:path json-path})))
        (let [model-json (js/JSON.parse (fs/readFileSync json-path "utf8"))
              manifest (or (gobj/get model-json "weightsManifest") #js [])
              groups (vec (array-seq manifest))
              specs (to-array (mapcat (fn [g] (array-seq (gobj/get g "weights"))) groups))
              buffers (to-array (mapcat (fn [g]
                                          (map (fn [rel]
                                                 (let [f (path/resolve dir rel)]
                                                   (when-not (fs/existsSync f)
                                                     (throw (util/io-error (str "weight shard not found: " f) {:path f})))
                                                   (node-buffer->array-buffer (fs/readFileSync f))))
                                               (array-seq (gobj/get g "paths"))))
                                        groups))
              weight-data ((gobj/get tf/io "concatenateArrayBuffers") buffers)]
          ((gobj/get tf/io "getModelArtifactsForJSONSync") model-json specs weight-data))
        (catch :default e
          (throw (if (some? (ex-data e)) e
                     (util/io-error (str "cannot read model from " dir ": " (gobj/get e "message")) {:dir dir :cause e}))))))))

(defn load-from-url!
  "Promise of a model fetched from an http(s) URL of a model.json (tfjs's own
   http handler); opts {:strict :weight-path-prefix :from-tf-hub ..}. Network
   failures reject with :tensorflow-cljs/io-error."
  [url opts]
  (async-guard #(tf/loadLayersModel url (->config-js (or opts {})))))

;; ---------------------------------------------------------------------------
;; Graph models (tfjs-converter, inventory §4.7): inference only
;; ---------------------------------------------------------------------------

(defn graph-model?
  "True for an opaque graph model (a converted SavedModel / Keras graph
   loaded with `load-graph-artifacts!`; inference only, never trainable)."
  [x]
  (instance? tf/GraphModel x))

(defn load-graph-artifacts!
  "Promise of an opaque graph model from opaque artifacts whose format is
   \"graph-model\" (a tensorflowjs_converter model.json + shards read with
   `read-artifacts-from-dir!`). Rejects with :tensorflow-cljs/io-error for
   layers artifacts (tfjs itself would crash reading `versions.producer`)."
  [artifacts]
  (if (= "graph-model" (artifacts-format artifacts))
    (async-guard #(tf/loadGraphModel ((gobj/get tf/io "fromMemory") artifacts)))
    (p/rejected (util/io-error (str "not a graph model (format " (pr-str (artifacts-format artifacts))
                                    "): use load! / load-artifacts! for layers models")
                               {:format (artifacts-format artifacts)}))))

(defn load-graph-from-url!
  "Promise of a graph model fetched from an http(s) URL of a model.json
   (tfjs's own http handler); opts {:from-tf-hub :weight-path-prefix ..}."
  [url opts]
  (async-guard #(tf/loadGraphModel url (->config-js (or opts {})))))

(defn graph-predict
  "Opaque output tensor (vector for multi-output graphs) of a graph model
   for `x`: a tensor, a vector of tensors in input-node order, or a map
   keyed by input name; opts {:batch-size :verbose}."
  [model x opts]
  (guard #(->result (js-invoke model "predict" (->js-container x) (->config-js (or opts {}))))))

(defn graph-execute
  "Tensor (one name) or vector of tensors of the graph nodes named in
   `output-names` for `x` (as in `graph-predict`). Synchronous: graphs with
   control-flow / dynamic-shape ops need `graph-execute!`."
  [model x output-names]
  (guard #(->result (js-invoke model "execute" (->js-container x) (->js-container output-names)))))

(defn graph-execute!
  "Promise form of `graph-execute` (executeAsync: required for graphs with
   control-flow or dynamic-shape ops)."
  [model x output-names]
  (-> (async-guard #(js-invoke model "executeAsync" (->js-container x) (->js-container output-names)))
      (p/then ->result)))

(defn- node-info [n]
  {:name (gobj/get n "name")
   :shape (let [s (gobj/get n "shape")] (when (array? s) (js-array->clj s)))
   :dtype (some-> (gobj/get n "dtype") util/str->dtype)})

(defn graph-inputs "Vector of {:name :shape :dtype} of the graph's input nodes (-1 = unknown dim)." [model] (mapv node-info (array-seq (gobj/get model "inputs"))))
(defn graph-outputs "Vector of {:name :shape :dtype} of the graph's output nodes." [model] (mapv node-info (array-seq (gobj/get model "outputs"))))
(defn graph-input-names "Vector of input node names." [model] (js-array->clj (gobj/get model "inputNodes")))
(defn graph-output-names "Vector of output node names." [model] (js-array->clj (gobj/get model "outputNodes")))
(defn graph-dispose! "Frees the graph model's weights; returns nil." [model] (guard #(js-invoke model "dispose")) nil)

;; ---------------------------------------------------------------------------
;; Data bridge (consumed by tensorflow-cljs.data)
;; ---------------------------------------------------------------------------

(defn- data-fn [name] (ns-fn tf/data name))

(defn- seq->js-iterator
  "JS iterator over a ClojureScript seq of elements (converted lazily)."
  [coll]
  (let [state (volatile! (seq coll))]
    #js {:next (fn []
                 (if-let [s @state]
                   (let [v (first s)]
                     (vreset! state (next s))
                     #js {:value (->js-container v) :done false})
                   #js {:value nil :done true}))}))

(defn dataset-array
  "Opaque dataset over a vector of elements (numbers, tensors, vectors, maps
   like {:xs .. :ys ..})."
  [items]
  (guard #((data-fn "array") (->js-container (vec items)))))

(defn dataset-generator
  "Opaque dataset whose elements come from calling the zero-argument fn
   `f` anew on each iteration; `f` returns a (possibly lazy) seq."
  [f]
  (guard #((data-fn "generator") (fn [] (seq->js-iterator (f))))))

(defn dataset-func
  "Opaque dataset pulling elements from the zero-argument fn `f` until it
   returns nil (it may also return a promise)."
  [f]
  (guard #((data-fn "func")
           (fn []
             (let [v (f)]
               (if (p/promise? v)
                 (p/then v (fn [v] (if (nil? v) #js {:value nil :done true} #js {:value (->js-container v) :done false})))
                 (if (nil? v) #js {:value nil :done true} #js {:value (->js-container v) :done false})))))))

(defn dataset-csv
  "Opaque CSV dataset from a URL or file path (\"file://\" added for local
   paths) and an already-translated config."
  [source config]
  (let [src (if (re-find #"^(https?|file)://" source) source (str "file://" (path/resolve source)))]
    (guard #((data-fn "csv") src (->config-js (or config {}))))))

(defn dataset-zip
  "Opaque dataset zipping a map or vector of datasets into elements of the
   same shape."
  [container]
  (guard #((data-fn "zip") (->js-container container))))

(defn dataset-batch "Batches `n` elements (stacked column-wise); `small-last?` (default true)." ([ds n] (guard #(js-invoke ds "batch" n))) ([ds n small-last?] (guard #(js-invoke ds "batch" n small-last?))))
(defn dataset-map "Maps a ClojureScript fn (element -> element) over the dataset (elements cross the boundary as kebab-free keyword maps / vectors / opaque tensors)." [ds f] (guard #(js-invoke ds "map" (fn [e] (->js-container (f (element->clj e)))))))
(defn dataset-map-async "Like dataset-map with `f` returning a promise." [ds f] (guard #(js-invoke ds "mapAsync" (fn [e] (p/then (p/promise (f (element->clj e))) ->js-container)))))
(defn dataset-filter "Keeps the elements for which `pred` is truthy (size becomes nil)." [ds pred] (guard #(js-invoke ds "filter" (fn [e] (boolean (pred (element->clj e)))))))
(defn dataset-shuffle "Windowed shuffle over `buffer-size`; `seed` (string) and `reshuffle-each-iteration?` optional." ([ds buffer-size] (guard #(js-invoke ds "shuffle" buffer-size))) ([ds buffer-size seed reshuffle?] (guard #(js-invoke ds "shuffle" buffer-size (undef-if-nil seed) (undef-if-nil reshuffle?)))))
(defn dataset-repeat "Repeats `count` times (nil = forever; size becomes ##Inf)." [ds count] (guard #(js-invoke ds "repeat" (undef-if-nil count))))
(defn dataset-take "First `n` elements." [ds n] (guard #(js-invoke ds "take" n)))
(defn dataset-skip "Drops the first `n` elements." [ds n] (guard #(js-invoke ds "skip" n)))
(defn dataset-prefetch "Prefetches `n` elements." [ds n] (guard #(js-invoke ds "prefetch" n)))
(defn dataset-concatenate "`ds` followed by `other`." [ds other] (guard #(js-invoke ds "concatenate" other)))
(defn dataset-size "Static element count: a number, nil (unknown, e.g. after filter) or ##Inf (unbounded repeat)." [ds] (let [s (gobj/get ds "size")] (if (nil? s) nil s)))
(defn dataset-to-vec! "Promise of a vector of every element (materialises everything; tensors inside are owned by the caller)." [ds] (-> (async-guard #(js-invoke ds "toArray")) (p/then element->clj)))
(defn dataset-for-each! "Promise resolving after calling `f` on every element (f may return a promise)." [ds f] (async-guard #(js-invoke ds "forEachAsync" (fn [e] (f (element->clj e))))))
(defn dataset-iterator! "Promise of an opaque lazy iterator." [ds] (async-guard #(js-invoke ds "iterator")))
(defn iterator-next! "Promise of {:value element :done bool} from an opaque iterator." [it] (-> (async-guard #(js-invoke it "next")) (p/then (fn [r] {:value (element->clj (gobj/get r "value")) :done (boolean (gobj/get r "done"))}))))
(defn csv-column-names! "Promise of the column-name vector of a CSV dataset." [ds] (-> (async-guard #(js-invoke ds "columnNames")) (p/then js-array->clj)))
(defn dataset-max-buffer-size "The shuffle / prefetch buffer cap (10000)." [] (gobj/get (data-fn "Dataset") "MAX_BUFFER_SIZE"))
