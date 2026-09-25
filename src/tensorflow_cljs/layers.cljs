(ns tensorflow-cljs.layers
  "Keras-style layer catalog of tfjs-layers (inventory §3): every one of the
   72 layer constructors as a ClojureScript fn taking a kebab config map,
   e.g. (dense {:units 4 :activation :relu :kernel-initializer :glorot-uniform
   :input-shape [2]}) — 66 distinct constructors plus the 10 pooling aliases. Keys become camelCase, keyword values become tfjs
   registry strings (:glorot-uniform -> \"glorotUniform\", :channels-last ->
   \"channelsLast\", :gelu-new -> \"gelu_new\"); initializer / regularizer /
   constraint values may also be kebab maps ({:type :random-normal :mean 0
   :stddev 0.05}) or opaque objects from `initializer` / `regularizer` /
   `constraint`.

   Layers, symbolic tensors and initializer-like objects are OPAQUE values;
   this namespace never touches JavaScript — everything goes through
   tensorflow-cljs.core."
  (:refer-clojure :exclude [flatten])
  (:require [clojure.string :as str]
            [tensorflow-cljs.core :as core]
            [tensorflow-cljs.util :as util]))

;; ---------------------------------------------------------------------------
;; Registries (kebab keyword -> tfjs identifier) — data, inventory §3.3
;; ---------------------------------------------------------------------------

(def activations
  "Activation identifiers accepted in layer configs (`:activation`)."
  {:elu "elu" :hard-sigmoid "hardSigmoid" :linear "linear" :relu "relu" :relu6 "relu6"
   :selu "selu" :sigmoid "sigmoid" :softmax "softmax" :softplus "softplus"
   :softsign "softsign" :tanh "tanh" :swish "swish" :mish "mish" :gelu "gelu"
   :gelu-new "gelu_new"})

(def initializers
  "Initializer identifiers (string form and `initializer` constructor names)."
  {:constant "constant" :glorot-normal "glorotNormal" :glorot-uniform "glorotUniform"
   :he-normal "heNormal" :he-uniform "heUniform" :identity "identity"
   :le-cun-normal "leCunNormal" :le-cun-uniform "leCunUniform" :ones "ones"
   :orthogonal "orthogonal" :random-normal "randomNormal" :random-uniform "randomUniform"
   :truncated-normal "truncatedNormal" :variance-scaling "varianceScaling" :zeros "zeros"})

(def regularizers
  "Regularizer constructor names (:l1l2 is also valid as a string identifier)."
  {:l1 "l1" :l2 "l2" :l1l2 "l1l2"})

(def constraints
  "Constraint identifiers."
  {:max-norm "maxNorm" :min-max-norm "minMaxNorm" :non-neg "nonNeg" :unit-norm "unitNorm"})

(def layer-aliases
  "The 10 alias exports of tf.layers (same constructors as the pooling
   layers under their long names)."
  {:avg-pool1d "avgPool1d" :avg-pool2d "avgPool2d" :avg-pool3d "avgPool3d"
   :avg-pooling1d "avgPooling1d" :avg-pooling2d "avgPooling2d" :avg-pooling3d "avgPooling3d"
   :max-pool1d "maxPool1d" :max-pool2d "maxPool2d"
   :global-max-pool1d "globalMaxPool1d" :global-max-pool2d "globalMaxPool2d"})

(def layer-constructors
  "Kebab layer name -> tf.layers constructor name: the 66 distinct
   constructors of tfjs-layers 4.22 plus the 10 `layer-aliases` (76 callable;
   `input` is separate)."
  (merge
   layer-aliases
   {:input-layer "inputLayer"
   ;; advanced activations
   :elu "elu" :relu "reLU" :leaky-relu "leakyReLU" :prelu "prelu" :softmax "softmax"
   :thresholded-relu "thresholdedReLU"
   ;; convolutional
   :conv1d "conv1d" :conv2d "conv2d" :conv2d-transpose "conv2dTranspose" :conv3d "conv3d"
   :conv3d-transpose "conv3dTranspose" :separable-conv2d "separableConv2d"
   :depthwise-conv2d "depthwiseConv2d" :cropping2d "cropping2D" :up-sampling2d "upSampling2d"
   ;; core
   :activation "activation" :dense "dense" :dropout "dropout" :spatial-dropout1d "spatialDropout1d"
   :flatten "flatten" :repeat-vector "repeatVector" :reshape "reshape" :permute "permute"
   :embedding "embedding" :masking "masking"
   ;; merge
   :add "add" :average "average" :concatenate "concatenate" :maximum "maximum"
   :minimum "minimum" :multiply "multiply" :dot "dot"
   ;; normalization
   :batch-normalization "batchNormalization" :layer-normalization "layerNormalization"
   ;; padding
   :zero-padding2d "zeroPadding2d"
   ;; pooling
   :average-pooling1d "averagePooling1d" :average-pooling2d "averagePooling2d"
   :average-pooling3d "averagePooling3d" :max-pooling1d "maxPooling1d"
   :max-pooling2d "maxPooling2d" :max-pooling3d "maxPooling3d"
   :global-average-pooling1d "globalAveragePooling1d" :global-average-pooling2d "globalAveragePooling2d"
   :global-max-pooling1d "globalMaxPooling1d" :global-max-pooling2d "globalMaxPooling2d"
   ;; recurrent
   :simple-rnn "simpleRNN" :gru "gru" :lstm "lstm" :simple-rnn-cell "simpleRNNCell"
   :gru-cell "gruCell" :lstm-cell "lstmCell" :conv-lstm2d "convLstm2d" :conv-lstm2d-cell "convLstm2dCell"
   :rnn "rnn" :stacked-rnn-cells "stackedRNNCells"
   ;; wrappers
   :bidirectional "bidirectional" :time-distributed "timeDistributed"
   ;; noise
   :gaussian-noise "gaussianNoise" :gaussian-dropout "gaussianDropout" :alpha-dropout "alphaDropout"
   ;; preprocessing
   :rescaling "rescaling" :center-crop "centerCrop" :resizing "resizing"
   :category-encoding "categoryEncoding" :random-width "randomWidth"}))

(defn- registry-name
  "Keyword -> registry string (argument-error when unknown); strings and
   opaque objects pass."
  [registry what k]
  (if (keyword? k)
    (or (get registry k)
        (throw (util/argument-error (str "unknown " what " " (pr-str k))
                                    {what k :allowed (set (keys registry))})))
    k))

;; ---------------------------------------------------------------------------
;; Initializer / regularizer / constraint objects
;; ---------------------------------------------------------------------------

(defn initializer
  "Opaque initializer: (initializer :random-normal {:mean 0 :stddev 0.05
   :seed 1}), (initializer :constant {:value 0.1}), (initializer
   :variance-scaling {:scale 1 :mode :fan-in :distribution :normal}),
   (initializer :glorot-uniform {:seed 3}) .."
  ([type] (initializer type nil))
  ([type args] (core/make-initializer (registry-name initializers "initializer" type) args)))

(defn regularizer
  "Opaque regularizer: (regularizer :l2 {:l2 0.01}), (regularizer :l1 {:l1
   0.01}), (regularizer :l1l2 {:l1 0.01 :l2 0.01})."
  ([type] (regularizer type nil))
  ([type args] (core/make-regularizer (registry-name regularizers "regularizer" type) args)))

(defn constraint
  "Opaque constraint: (constraint :max-norm {:max-value 2 :axis 0}),
   (constraint :unit-norm {:axis 0}), (constraint :non-neg), (constraint
   :min-max-norm {:min-value 0 :max-value 1 :rate 1})."
  ([type] (constraint type nil))
  ([type args] (core/make-constraint (registry-name constraints "constraint" type) args)))

(defn- object-value
  "A map {:type :random-normal ..} under an -initializer / -regularizer /
   -constraint key becomes the opaque object; keywords are validated."
  [k v]
  (let [n (name k)]
    (cond
      (str/ends-with? n "-initializer")
      (cond (map? v) (initializer (:type v) (dissoc v :type))
            (keyword? v) (registry-name initializers "initializer" v)
            :else v)
      (str/ends-with? n "-regularizer")
      (cond (map? v) (regularizer (:type v) (dissoc v :type))
            (keyword? v) (registry-name regularizers "regularizer" v)
            :else v)
      (str/ends-with? n "-constraint")
      (cond (map? v) (constraint (:type v) (dissoc v :type))
            (keyword? v) (registry-name constraints "constraint" v)
            :else v)
      (= k :activation)
      (registry-name activations "activation" v)
      (= k :recurrent-activation)
      (registry-name activations "activation" v)
      :else v)))

(defn prepare-config
  "Layer config -> config ready for the core bridge: activation keywords
   validated against `activations`, initializer / regularizer / constraint
   keywords validated and maps constructed. Pure."
  [config]
  (into {} (map (fn [[k v]] [k (object-value k v)])) config))

(defn layer
  "Generic constructor: (layer :dense {:units 4}) for any key of
   `layer-constructors`."
  ([type] (layer type nil))
  ([type config]
   (core/make-layer (registry-name layer-constructors "layer" type) (prepare-config (or config {})))))

(defn- ctor [type]
  (fn ([] (layer type nil)) ([config] (layer type config))))

;; ---------------------------------------------------------------------------
;; The catalog (one fn per constructor; config keys per inventory §3.1).
;; Common base keys: :input-shape :batch-input-shape :batch-size :dtype :name
;; :trainable :weights (vector of tensors) :input-d-type
;; ---------------------------------------------------------------------------

(def input-layer "Explicit input layer {:input-shape [..] | :batch-input-shape [..] :dtype :sparse :name}." (ctor :input-layer))

;; advanced activations
(def elu "ELU activation layer {:alpha 1}." (ctor :elu))
(def relu "ReLU activation layer {:max-value n}." (ctor :relu))
(def leaky-relu "Leaky ReLU layer {:alpha 0.3}." (ctor :leaky-relu))
(def prelu "Parametric ReLU layer {:alpha-initializer :alpha-regularizer :alpha-constraint :shared-axes}." (ctor :prelu))
(def softmax "Softmax layer {:axis -1}." (ctor :softmax))
(def thresholded-relu "Thresholded ReLU layer {:theta 1}." (ctor :thresholded-relu))

;; convolutional: {:filters :kernel-size :strides :padding :valid|:same|:causal :data-format :channels-last|:channels-first :dilation-rate :activation :use-bias :kernel-initializer :bias-initializer :kernel-constraint :bias-constraint :kernel-regularizer :bias-regularizer :activity-regularizer}
(def conv1d "1-D convolution layer {:filters n :kernel-size k :strides :padding :activation ..}." (ctor :conv1d))
(def conv2d "2-D convolution layer {:filters n :kernel-size [h w]|k :strides :padding :activation :input-shape [h w c] ..}." (ctor :conv2d))
(def conv2d-transpose "Transposed 2-D convolution layer (conv2d keys)." (ctor :conv2d-transpose))
(def conv3d "3-D convolution layer (conv2d keys)." (ctor :conv3d))
(def conv3d-transpose "Transposed 3-D convolution layer (conv2d keys)." (ctor :conv3d-transpose))
(def separable-conv2d "Depthwise-separable 2-D convolution {:filters :kernel-size :depth-multiplier :depthwise-initializer :pointwise-initializer ..}." (ctor :separable-conv2d))
(def depthwise-conv2d "Depthwise 2-D convolution {:kernel-size :depth-multiplier :depthwise-initializer ..}." (ctor :depthwise-conv2d))
(def cropping2d "Crops [[top bottom] [left right]] from images {:cropping :data-format}." (ctor :cropping2d))
(def up-sampling2d "Repeats rows / columns {:size [2 2] :interpolation :nearest|:bilinear :data-format}." (ctor :up-sampling2d))

;; core
(def activation "Applies an activation {:activation :relu}." (ctor :activation))
(def dense "Fully connected layer {:units n :activation :use-bias :kernel-initializer :bias-initializer :input-dim :kernel-regularizer :bias-regularizer :activity-regularizer :kernel-constraint :bias-constraint}." (ctor :dense))
(def dropout "Dropout {:rate r :noise-shape :seed}." (ctor :dropout))
(def spatial-dropout1d "Drops whole 1-D feature maps {:rate :seed}." (ctor :spatial-dropout1d))
(def flatten "Flattens all but the batch dimension {:data-format}." (ctor :flatten))
(def repeat-vector "Repeats the input n times {:n}." (ctor :repeat-vector))
(def reshape "Reshapes to {:target-shape [..]}." (ctor :reshape))
(def permute "Permutes dimensions {:dims [2 1]} (1-based, batch excluded)." (ctor :permute))
(def embedding "Integer indices -> dense vectors {:input-dim vocab :output-dim size :embeddings-initializer :embeddings-regularizer :activity-regularizer :embeddings-constraint :mask-zero :input-length}." (ctor :embedding))
(def masking "Masks timesteps equal to {:mask-value 0}." (ctor :masking))

;; merge (applied to a VECTOR of inputs)
(def add "Elementwise sum of a vector of inputs." (ctor :add))
(def average "Elementwise average of a vector of inputs." (ctor :average))
(def concatenate "Concatenates a vector of inputs along {:axis -1}." (ctor :concatenate))
(def maximum "Elementwise maximum of a vector of inputs." (ctor :maximum))
(def minimum "Elementwise minimum of a vector of inputs." (ctor :minimum))
(def multiply "Elementwise product of a vector of inputs." (ctor :multiply))
(def dot "Dot product of two inputs along {:axes n|[a b] :normalize false}." (ctor :dot))

;; normalization
(def batch-normalization "Batch normalisation {:axis -1 :momentum 0.99 :epsilon 0.001 :center :scale :beta-initializer :gamma-initializer :moving-mean-initializer :moving-variance-initializer :beta-constraint :gamma-constraint :beta-regularizer :gamma-regularizer}." (ctor :batch-normalization))
(def layer-normalization "Layer normalisation {:axis -1 :epsilon 0.001 :center :scale :beta-initializer :gamma-initializer :beta-regularizer :gamma-regularizer}." (ctor :layer-normalization))

;; padding
(def zero-padding2d "Zero-pads images {:padding n|[h w]|[[t b] [l r]] :data-format}." (ctor :zero-padding2d))

;; pooling: {:pool-size :strides :padding :data-format}
(def average-pooling1d "1-D average pooling {:pool-size :strides :padding}." (ctor :average-pooling1d))
(def average-pooling2d "2-D average pooling {:pool-size [2 2] :strides :padding :data-format}." (ctor :average-pooling2d))
(def average-pooling3d "3-D average pooling." (ctor :average-pooling3d))
(def max-pooling1d "1-D max pooling {:pool-size :strides :padding}." (ctor :max-pooling1d))
(def max-pooling2d "2-D max pooling {:pool-size [2 2] :strides :padding :data-format}." (ctor :max-pooling2d))
(def max-pooling3d "3-D max pooling." (ctor :max-pooling3d))
(def global-average-pooling1d "Averages over the time dimension." (ctor :global-average-pooling1d))
(def global-average-pooling2d "Averages over the spatial dimensions {:data-format}." (ctor :global-average-pooling2d))
(def global-max-pooling1d "Max over the time dimension." (ctor :global-max-pooling1d))
(def global-max-pooling2d "Max over the spatial dimensions {:data-format}." (ctor :global-max-pooling2d))

;; recurrent: {:units :activation :recurrent-activation :use-bias :kernel-initializer :recurrent-initializer :bias-initializer :kernel-regularizer :recurrent-regularizer :bias-regularizer :kernel-constraint :recurrent-constraint :bias-constraint :dropout :recurrent-dropout :return-sequences :return-state :go-backwards :stateful :unroll :input-dim :input-length}
(def simple-rnn "Fully connected RNN {:units :activation :return-sequences ..}." (ctor :simple-rnn))
(def gru "Gated recurrent unit {:units :recurrent-activation :reset-after :implementation ..}." (ctor :gru))
(def lstm "Long short-term memory {:units :recurrent-activation :unit-forget-bias :implementation :return-sequences :return-state ..}." (ctor :lstm))
(def simple-rnn-cell "SimpleRNN cell for `rnn` / `stacked-rnn-cells` {:units ..}." (ctor :simple-rnn-cell))
(def gru-cell "GRU cell {:units ..}." (ctor :gru-cell))
(def lstm-cell "LSTM cell {:units ..}." (ctor :lstm-cell))
(def conv-lstm2d "Convolutional LSTM {:filters :kernel-size :strides :padding :return-sequences ..}." (ctor :conv-lstm2d))
(def conv-lstm2d-cell "ConvLSTM2D cell." (ctor :conv-lstm2d-cell))
(def rnn "Generic RNN over {:cell cell|[cells] :return-sequences :return-state :go-backwards :stateful :unroll :input-dim :input-length}." (ctor :rnn))
(def stacked-rnn-cells "Stacks {:cells [..]} into one cell." (ctor :stacked-rnn-cells))

;; wrappers
(def bidirectional "Runs {:layer rnn-layer} forwards and backwards {:merge-mode :concat|:sum|:mul|:ave|nil}." (ctor :bidirectional))
(def time-distributed "Applies {:layer l} to every timestep." (ctor :time-distributed))

;; noise
(def gaussian-noise "Additive zero-mean Gaussian noise {:stddev}." (ctor :gaussian-noise))
(def gaussian-dropout "Multiplicative Gaussian noise {:rate}." (ctor :gaussian-dropout))
(def alpha-dropout "SELU-preserving dropout {:rate :noise-shape}." (ctor :alpha-dropout))

;; preprocessing
(def rescaling "x * scale + offset {:scale :offset 0}." (ctor :rescaling))
(def center-crop "Central crop to {:height :width}." (ctor :center-crop))
(def resizing "Resizes images to {:height :width :interpolation :bilinear|:nearest :crop-to-aspect-ratio false}." (ctor :resizing))
(def category-encoding "Integer -> categorical encoding {:num-tokens :output-mode :multi-hot|:one-hot|:count}." (ctor :category-encoding))
(def random-width "Randomly varies image width {:factor :interpolation :seed :auto-vectorize}." (ctor :random-width))

;; ---------------------------------------------------------------------------
;; Symbolic inputs and application
;; ---------------------------------------------------------------------------

(defn input
  "Symbolic input for a functional model: (input {:shape [4] :name \"x\"
   :dtype :float32}) or {:batch-shape [32 4]}. The batch dimension is nil."
  [config]
  (core/make-input config))

(defn apply-layer
  "Applies `layer` to `inputs` (one value or a vector): symbolic in ->
   symbolic out (graph building); tensor in -> tensor out (eager). Optional
   kebab kwargs (e.g. {:training true})."
  ([layer inputs] (core/layer-apply layer inputs))
  ([layer inputs kwargs] (core/layer-apply layer inputs kwargs)))

;; ---------------------------------------------------------------------------
;; Accessors
;; ---------------------------------------------------------------------------

(def layer? "True for an opaque layer." core/layer?)
(def symbolic? "True for a symbolic tensor." core/symbolic?)
(defn weights "Vector of the layer's weight tensors (kernel, bias ..); (weights l true) for trainable only." ([layer] (core/layer-weights layer)) ([layer trainable-only?] (core/layer-weights layer trainable-only?)))
(defn set-weights! "Replaces the weights from a vector of tensors (same shapes / order); returns the layer." [layer tensors] (core/layer-set-weights! layer tensors))
(defn count-params "Number of scalar parameters (layer must be built)." [layer] (core/layer-count-params layer))
(defn config "Kebab map of the layer's configuration ({:units 4 :activation \"relu\" ..})." [layer] (core/layer-config layer))
(defn layer-name "Unique name (\"dense_Dense1\")." [layer] (core/layer-name layer))
(defn class-name "Keras class name (\"Dense\")." [layer] (core/layer-class-name layer))
(defn trainable? "True when the layer's weights train." [layer] (core/layer-trainable? layer))
(defn set-trainable! "Freezes / unfreezes; returns the layer." [layer trainable?] (core/layer-set-trainable! layer trainable?))
(defn built? "True once weights exist." [layer] (core/layer-built? layer))
(defn build! "Creates the weights for `input-shape` (nil = batch dim); returns the layer." [layer input-shape] (core/layer-build! layer input-shape))
(defn output-shape "Output shape (nil = batch dim) of a built layer." [layer] (core/layer-output-shape layer))
(defn compute-output-shape "Output shape for `input-shape` (nil = batch dim)." [layer input-shape] (core/layer-compute-output-shape layer input-shape))
(defn batch-input-shape "Batch input shape of an input-carrying layer, or nil." [layer] (core/layer-batch-input-shape layer))
(defn dispose! "Frees the layer's weights; returns {:ref-count-after-dispose n :num-disposed-variables n}." [layer] (core/layer-dispose! layer))
(defn reset-states! "Resets a stateful RNN's states; returns the layer." [layer] (core/layer-reset-states! layer))
(defn symbolic-shape "Shape of a symbolic tensor (nil = batch dim)." [s] (core/symbolic-shape s))
(defn symbolic-name "Name of a symbolic tensor." [s] (core/symbolic-name s))
