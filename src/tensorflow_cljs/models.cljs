(ns tensorflow-cljs.models
  "Models (inventory §4, §6): sequential and functional construction,
   compile / fit / predict / evaluate, tf.data training, callbacks, save /
   load, weights and summaries. Models are OPAQUE values; configs are
   kebab keyword maps; asynchronous tfjs calls return promesa promises
   (suffix `!`: fit!, fit-dataset!, evaluate-dataset!, train-on-batch!,
   save!, load!, predict!, evaluate!); `predict` and `evaluate` are also
   offered synchronously as tfjs itself runs them.

   No JavaScript interop here — everything goes through tensorflow-cljs.core."
  (:refer-clojure :exclude [sequential?])
  (:require [clojure.string :as str]
            [promesa.core :as p]
            [tensorflow-cljs.core :as core]
            [tensorflow-cljs.util :as util]))

;; ---------------------------------------------------------------------------
;; Registries (kebab keyword -> compile-time tfjs identifier), inventory §3.3
;; ---------------------------------------------------------------------------

(def losses
  "Loss identifiers accepted by `compile-model!` (camelCase strings in tfjs;
   :mse :mae :mape :msle :kld :cosine are aliases of the long names)."
  {:mean-squared-error "meanSquaredError" :mse "meanSquaredError"
   :mean-absolute-error "meanAbsoluteError" :mae "meanAbsoluteError"
   :mean-absolute-percentage-error "meanAbsolutePercentageError" :mape "meanAbsolutePercentageError"
   :mean-squared-logarithmic-error "meanSquaredLogarithmicError" :msle "meanSquaredLogarithmicError"
   :squared-hinge "squaredHinge" :hinge "hinge" :categorical-hinge "categoricalHinge"
   :logcosh "logcosh" :categorical-crossentropy "categoricalCrossentropy"
   :sparse-categorical-crossentropy "sparseCategoricalCrossentropy"
   :binary-crossentropy "binaryCrossentropy"
   :kullback-leibler-divergence "kullbackLeiblerDivergence" :kld "kullbackLeiblerDivergence"
   :poisson "poisson" :cosine-proximity "cosineProximity" :cosine "cosineProximity"})

(def metrics
  "Metric identifiers accepted by `compile-model!` (:accuracy / :acc resolve
   per output to binary / categorical / sparse-categorical accuracy from the
   loss); every loss identifier is also accepted."
  (merge losses
         {:accuracy "accuracy" :acc "acc" :crossentropy "crossentropy" :ce "ce"
          :binary-accuracy "binaryAccuracy" :categorical-accuracy "categoricalAccuracy"
          :precision "precision" :mse "mse" :mae "mae" :mape "mape" :cosine "cosine"}))

(def optimizers
  "Optimizer identifiers accepted by `compile-model!` as bare keywords (tfjs
   defaults: sgd lr 0.01, adam 0.001, adamax 0.002, rmsprop 0.001, adadelta
   1, adagrad 0.01) and by `optimizer` with a kebab option map."
  {:sgd "sgd" :adagrad "adagrad" :adadelta "adadelta" :adam "adam" :adamax "adamax" :rmsprop "rmsprop"
   :momentum "momentum"})

(def optimizer-args
  "Positional constructor arguments of tf.train.* per optimizer, in order,
   with the defaults used when the option map omits them."
  {:sgd [[:learning-rate 0.01]]
   :momentum [[:learning-rate 0.01] [:momentum 0.9] [:use-nesterov nil]]
   :rmsprop [[:learning-rate 0.001] [:decay nil] [:momentum nil] [:epsilon nil] [:centered nil]]
   :adam [[:learning-rate 0.001] [:beta1 nil] [:beta2 nil] [:epsilon nil]]
   :adadelta [[:learning-rate 1] [:rho nil] [:epsilon nil]]
   :adamax [[:learning-rate 0.002] [:beta1 nil] [:beta2 nil] [:epsilon nil] [:decay nil]]
   :adagrad [[:learning-rate 0.01] [:initial-accumulator-value nil]]})

(defn- registry-name [registry what k]
  (if (keyword? k)
    (or (get registry k)
        (throw (util/argument-error (str "unknown " what " " (pr-str k)) {what k :allowed (set (keys registry))})))
    k))

(defn optimizer
  "Opaque optimizer with explicit hyper-parameters: (optimizer :adam
   {:learning-rate 0.01 :beta1 0.9}), (optimizer :momentum {:learning-rate
   0.1 :momentum 0.9 :use-nesterov true}), (optimizer :sgd). Keys per
   `optimizer-args`."
  ([type] (optimizer type nil))
  ([type opts]
   (let [spec (or (get optimizer-args type)
                  (throw (util/argument-error (str "unknown optimizer " (pr-str type)) {:optimizer type :allowed (set (keys optimizer-args))})))
         args (map (fn [[k default]] (get opts k default)) spec)]
     (core/make-optimizer (get optimizers type) args))))

(defn- optimizer-arg
  "compile :optimizer -> tfjs value: keyword (registry string), {:type :adam
   :learning-rate ..} map (constructed), string or opaque optimizer (as is)."
  [o]
  (cond (map? o) (optimizer (:type o) (dissoc o :type))
        (keyword? o) (registry-name optimizers "optimizer" o)
        :else o))

(defn- loss-arg [l]
  (cond (keyword? l) (registry-name losses "loss" l)
        (map? l) (into {} (map (fn [[k v]] [(name k) (loss-arg v)])) l)
        (cljs.core/sequential? l) (mapv loss-arg l)
        :else l))

(defn- metric-arg [m]
  (cond (keyword? m) (registry-name metrics "metric" m)
        (map? m) (into {} (map (fn [[k v]] [(name k) (metric-arg v)])) m)
        (cljs.core/sequential? m) (mapv metric-arg m)
        :else m))

;; ---------------------------------------------------------------------------
;; Construction
;; ---------------------------------------------------------------------------

(defn sequential
  "Sequential model from a vector of layers (the first carries
   :input-shape); opts {:name ..}. (sequential) is an empty model to fill
   with `add-layer!`."
  ([] (sequential [] nil))
  ([layers] (sequential layers nil))
  ([layers {:keys [name]}] (core/make-sequential layers name)))

(defn make-sequential
  "Alias of `sequential`."
  ([] (sequential))
  ([layers] (sequential layers))
  ([layers opts] (sequential layers opts)))

(defn add-layer!
  "Appends a layer to a sequential model; returns the model."
  [model layer]
  (core/model-add! model layer))

(defn pop-layer!
  "Removes the last layer of a sequential model; returns the model."
  [model]
  (core/model-pop! model))

(defn model
  "Functional model from a graph of symbolic tensors: {:inputs s | [s ..]
   :outputs s | [s ..] :name ..} where the outputs were built with
   tensorflow-cljs.layers/input and apply-layer."
  [{:keys [inputs outputs name] :as config}]
  (when (or (nil? inputs) (nil? outputs))
    (throw (util/argument-error "model needs :inputs and :outputs" {:config (keys config)})))
  (core/make-model {:inputs inputs :outputs outputs :name name}))

(def make-model "Alias of `model`." model)

;; ---------------------------------------------------------------------------
;; Compile / fit / predict / evaluate
;; ---------------------------------------------------------------------------

(defn compile-model!
  "Compiles the model: {:optimizer :adam | {:type :adam :learning-rate 0.01}
   | opaque, :loss :mean-squared-error | (fn [y-true y-pred] tensor) | vector
   | {output-name loss}, :metrics [:accuracy :mae] | {output-name [..]}}.
   Keywords are translated through `optimizers` / `losses` / `metrics`;
   unknown ones throw :tensorflow-cljs/argument-error before tfjs is
   called. Synchronous; returns the model."
  [model {:keys [optimizer loss metrics] :as config}]
  (when (nil? optimizer) (throw (util/argument-error "compile-model! needs :optimizer" {:config config})))
  (when (nil? loss) (throw (util/argument-error "compile-model! needs :loss" {:config config})))
  (core/model-compile! model (cond-> {:optimizer (optimizer-arg optimizer) :loss (loss-arg loss)}
                               (some? metrics) (assoc :metrics (metric-arg metrics)))))

(def compile! "Alias of `compile-model!`." compile-model!)

(defn fit!
  "Promise of the training history {:epoch [0 1 ..] :history {:loss [..]
   :acc [..] :val-loss [..]} :params {:epochs n :batch-size n ..}} for
   tensors (or vectors / maps of tensors) `x` / `y`. Options (kebab, tfjs
   defaults): :epochs 1, :batch-size 32, :shuffle true, :validation-split
   0..1, :validation-data [xv yv], :verbose 0 (wrapper default; 1 prints a
   progress bar), :callbacks {:on-epoch-end (fn [epoch logs]) ..} or a
   vector of such maps and / or `early-stopping` values, :class-weight
   {class weight}, :sample-weight t, :initial-epoch, :steps-per-epoch,
   :yield-every :auto|:batch|:epoch|:never|ms. A second fit! on a model
   while one runs rejects (tfjs guard)."
  ([model x y] (fit! model x y nil))
  ([model x y opts]
   (core/model-fit! model x y (merge {:verbose 0} opts))))

(defn fit-dataset!
  "Promise of the history of training on an opaque tf.data dataset whose
   elements are {:xs .. :ys ..} batches (see tensorflow-cljs.data). Options:
   :epochs, :batches-per-epoch (REQUIRED when the dataset size is unknown or
   infinite — :argument-error otherwise), :callbacks, :validation-data
   (dataset or [xs ys]), :validation-batches, :validation-batch-size,
   :yield-every, :initial-epoch, :class-weight, :verbose 0."
  ([model dataset] (fit-dataset! model dataset nil))
  ([model dataset opts]
   (let [size (core/dataset-size dataset)]
     (if (and (or (nil? size) (infinite? size)) (nil? (:batches-per-epoch opts)))
       (p/rejected (util/argument-error "fit-dataset!: the dataset size is unknown / infinite, pass :batches-per-epoch"
                                        {:size size}))
       (core/model-fit-dataset! model dataset (merge {:verbose 0} opts))))))

(defn train-on-batch!
  "Promise of the loss (number) or [loss metric ..] of one gradient step on
   a batch `x` / `y`."
  [model x y]
  (core/model-train-on-batch! model x y))

(defn predict
  "Opaque output tensor (vector of tensors for multi-output models) for `x`
   (tensor / vector / map); opts {:batch-size 32}. Synchronous."
  ([model x] (predict model x nil))
  ([model x opts] (core/model-predict model x opts)))

(defn predict!
  "Promise form of `predict` (rejects instead of throwing)."
  ([model x] (predict! model x nil))
  ([model x opts] (p/do (predict model x opts))))

(defn predict-on-batch
  "Opaque output for one batch `x` without internal batching. Synchronous."
  [model x]
  (core/model-predict-on-batch model x))

(defn predict-on-batch!
  "Promise form of `predict-on-batch`."
  [model x]
  (p/do (predict-on-batch model x)))

(defn evaluate
  "Kebab map metric -> number ({:loss 0.1 :acc 1}) of the compiled model on
   `x` / `y`; opts {:batch-size :sample-weight :steps}. Synchronous; the
   tfjs scalars are read and disposed."
  ([model x y] (evaluate model x y nil))
  ([model x y opts] (core/model-evaluate model x y opts)))

(defn evaluate!
  "Promise form of `evaluate`."
  ([model x y] (evaluate! model x y nil))
  ([model x y opts] (p/do (evaluate model x y opts))))

(defn evaluate-dataset!
  "Promise of the kebab metrics map over an opaque dataset of {:xs :ys}
   batches; opts {:batches n}."
  ([model dataset] (evaluate-dataset! model dataset nil))
  ([model dataset opts] (core/model-evaluate-dataset! model dataset opts)))

(defn execute
  "Vector of intermediate output tensors of the layers named in
   `output-names` for input `x`."
  [model x output-names]
  (core/model-execute model x output-names))

;; ---------------------------------------------------------------------------
;; Callbacks
;; ---------------------------------------------------------------------------

(defn callback
  "Opaque callback from a hook map {:on-epoch-end (fn [epoch logs]) ..}
   (fit! accepts the map directly; use this to mix with `early-stopping`)."
  [hooks]
  (core/make-callback hooks))

(defn- monitor-name [k] (if (keyword? k) (str/replace (name k) "-" "_") k))

(defn early-stopping
  "Opaque EarlyStopping callback for fit!'s :callbacks vector: {:monitor
   :val-loss (default) :min-delta 0 :patience 0 :mode :auto|:min|:max
   :baseline n :restore-best-weights false :verbose 0}. The monitored key
   is a history key (:loss :val-loss :acc ..)."
  ([] (early-stopping nil))
  ([{:keys [monitor] :as opts}]
   (core/early-stopping (cond-> (or opts {}) (some? monitor) (assoc :monitor (monitor-name monitor))))))

(defn stop-training!
  "Asks the running fit! to stop after the current epoch (from a callback)."
  [model]
  (core/model-stop-training! model))

;; ---------------------------------------------------------------------------
;; Introspection / weights
;; ---------------------------------------------------------------------------

(def model? "True for an opaque model." core/model?)
(def sequential? "True for an opaque sequential model." core/sequential-model?)
(defn model-name "The model's name." [model] (core/model-name model))
(defn layers "Vector of the model's opaque layers in order." [model] (core/model-layers model))
(defn get-layer "Layer by name (string) or index (number, 0-based); :value-error when absent." [model name-or-index] (core/model-get-layer model name-or-index))
(defn weights "Vector of every weight tensor (kernel, bias .. per layer); (weights m true) for trainable only." ([model] (core/model-weights model)) ([model trainable-only?] (core/model-weights model trainable-only?)))
(defn set-weights! "Replaces every weight from a vector of tensors (shapes / order as `weights`); returns the model." [model tensors] (core/model-set-weights! model tensors))
(defn count-params "Total parameter count." [model] (core/model-count-params model))
(defn metrics-names "Vector of metric names after compile ([\"loss\" \"acc\"]), nil before." [model] (core/model-metrics-names model))
(defn compiled? "True once compile-model! ran (or a model with training config was loaded)." [model] (core/model-compiled? model))
(defn inputs "Vector of symbolic inputs." [model] (core/model-inputs model))
(defn outputs "Vector of symbolic outputs." [model] (core/model-outputs model))
(defn input-names "Vector of input names." [model] (core/model-input-names model))
(defn output-names "Vector of output names." [model] (core/model-output-names model))
(defn summary
  "The model summary as one string (the lines tfjs would print); opts
   {:line-length n :positions [..]}."
  ([model] (summary model nil))
  ([model opts] (str/join "\n" (core/model-summary model (or opts {})))))
(defn summary-lines "The summary as a vector of lines." ([model] (summary-lines model nil)) ([model opts] (core/model-summary model (or opts {}))))
(defn print-summary "Prints `summary`; returns nil." [model] (println (summary model)))
(defn dispose! "Frees the model's weights; returns {:ref-count-after-dispose n :num-disposed-variables n}." [model] (core/model-dispose! model))
(defn user-metadata "The metadata map attached with set-user-metadata! (string keys), or nil." [model] (core/model-user-metadata model))
(defn set-user-metadata! "Attaches a JSON-able map saved with the model; returns the model." [model m] (core/model-set-user-metadata! model m))

;; ---------------------------------------------------------------------------
;; Topology JSON and save / load
;; ---------------------------------------------------------------------------

(defn to-json
  "The Keras topology JSON as ClojureScript data (string keys verbatim)."
  [model]
  (core/model-to-json model))

(defn model-from-json!
  "Promise of a model rebuilt (fresh weights) from `to-json` data."
  [topology]
  (core/model-from-json! topology))

(defn save-artifacts!
  "Promise of opaque in-memory artifacts (topology + weights) of the model;
   opts {:include-optimizer false :trainable-only false}. Pair with
   `load-artifacts!` for an in-process round trip, or `artifacts-info` /
   `weight-specs` for inspection."
  ([model] (save-artifacts! model nil))
  ([model opts] (core/model-artifacts! model opts)))

(defn load-artifacts!
  "Promise of a model from opaque artifacts."
  [artifacts]
  (core/load-artifacts! artifacts))

(defn artifacts-info "{:date-saved .. :model-topology-bytes n :weight-data-bytes n ..} of artifacts." [artifacts] (core/artifacts-info artifacts))
(defn weight-specs "Vector of {:name :shape :dtype} weight specs of artifacts." [artifacts] (core/artifacts-weight-specs artifacts))

(defn save!
  "Saves the model to a directory (or a model.json path, or a \"file://\"
   URL) as `model.json` + one `weights.bin` shard — the tfjs-node /
   tensorflowjs_converter layout — with the library's own node-fs handler
   (pure tfjs has no file:// handler, inventory §4.6). opts
   {:include-optimizer false (true stores the training config so `load!`
   yields a compiled model) :trainable-only false}. Promise of {:dir
   :model-json :weights-bin :model-artifacts-info}."
  ([model path] (save! model path nil))
  ([model path opts]
   (p/let [artifacts (save-artifacts! model opts)]
     (core/save-artifacts-to-dir! artifacts path))))

(defn load!
  "Promise of a model from a directory / model.json path / \"file://\" URL
   written by `save!` (or tfjs-node / the converter: multi-shard manifests
   are concatenated in manifest order, quantized weights decoded by tfjs),
   or from an http(s) URL (tfjs's own fetch handler; opts {:strict
   :weight-path-prefix ..}). Rejects with :tensorflow-cljs/io-error when
   files are missing."
  ([path] (load! path nil))
  ([path opts]
   (if (re-find #"^https?://" path)
     (core/load-from-url! path opts)
     (p/let [artifacts (core/read-artifacts-from-dir! path)]
       (core/load-artifacts! artifacts)))))
