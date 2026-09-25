(ns tensorflow-cljs.layers-test
  "Layer catalog: config translation, eager application, weights, graphs."
  (:require [cljs.test :refer [deftest is testing]]
            [tensorflow-cljs.core :as tf]
            [tensorflow-cljs.layers :as layers]
            [tensorflow-cljs.test-util :as tu :refer [approx= thrown]]
            [tensorflow-cljs.util :as util]))

(deftest dense-config-translation
  (let [l (layers/dense {:units 3 :activation :relu :use-bias false :name "d1"})
        c (layers/config l)]
    (is (layers/layer? l))
    (is (= 3 (:units c)))
    (is (= "relu" (:activation c)) ":relu rendered as the registry string")
    (is (false? (:use-bias c)))
    (is (= "d1" (:name c)))
    (is (= "d1" (layers/layer-name l)))
    (is (= "Dense" (layers/class-name l)))
    (is (= "VarianceScaling" (get-in c [:kernel-initializer :class-name])) "default glorot uniform initializer")
    (is (not (layers/built? l)))))

(deftest every-constructor-builds
  (let [cell #(layers/simple-rnn-cell {:units 1})
        configs {:input-layer {:input-shape [1]}
                 :elu {} :relu {} :leaky-relu {} :prelu {} :softmax {} :thresholded-relu {}
                 :conv1d {:filters 1 :kernel-size 1} :conv2d {:filters 1 :kernel-size 1}
                 :conv2d-transpose {:filters 1 :kernel-size 1} :conv3d {:filters 1 :kernel-size 1}
                 :conv3d-transpose {:filters 1 :kernel-size 1} :separable-conv2d {:filters 1 :kernel-size 1}
                 :depthwise-conv2d {:kernel-size 1} :cropping2d {:cropping 1} :up-sampling2d {}
                 :activation {:activation :tanh} :dense {:units 1} :dropout {:rate 0.1}
                 :spatial-dropout1d {:rate 0.1} :flatten {} :repeat-vector {:n 2} :reshape {:target-shape [1]}
                 :permute {:dims [1]} :embedding {:input-dim 2 :output-dim 2} :masking {}
                 :add {} :average {} :concatenate {} :maximum {} :minimum {} :multiply {} :dot {:axes 1}
                 :batch-normalization {} :layer-normalization {} :zero-padding2d {}
                 :average-pooling1d {} :average-pooling2d {} :average-pooling3d {}
                 :max-pooling1d {} :max-pooling2d {} :max-pooling3d {}
                 :global-average-pooling1d {} :global-average-pooling2d {}
                 :global-max-pooling1d {} :global-max-pooling2d {}
                 :simple-rnn {:units 1} :gru {:units 1} :lstm {:units 1}
                 :simple-rnn-cell {:units 1} :gru-cell {:units 1} :lstm-cell {:units 1}
                 :conv-lstm2d {:filters 1 :kernel-size 1} :conv-lstm2d-cell {:filters 1 :kernel-size 1}
                 :rnn {:cell (cell)} :stacked-rnn-cells {:cells [(cell) (cell)]}
                 :bidirectional {:layer (layers/lstm {:units 1})} :time-distributed {:layer (layers/dense {:units 1})}
                 :gaussian-noise {:stddev 1} :gaussian-dropout {:rate 0.1} :alpha-dropout {:rate 0.1}
                 :rescaling {:scale 1} :center-crop {:height 1 :width 1} :resizing {:height 1 :width 1}
                 :category-encoding {:num-tokens 2} :random-width {:factor 0.1}}
        built (into {} (map (fn [[k cfg]] [k (layers/layer k cfg)])) configs)]
    (is (= (set (keys layers/layer-constructors)) (into (set (keys configs)) (keys layers/layer-aliases)))
        "every catalogued constructor is exercised")
    (is (= 66 (count configs)) "66 distinct constructors")
    (is (= 10 (count layers/layer-aliases)))
    (is (= 76 (count layers/layer-constructors)))
    (is (= "AveragePooling2D" (layers/class-name (layers/layer :avg-pool2d))) "aliases build the long-name layers")
    (is (= "GlobalMaxPooling1D" (layers/class-name (layers/layer :global-max-pool1d))))
    (is (every? layers/layer? (vals built)))
    (is (= "Dense" (layers/class-name (:dense built))))
    (is (= "LSTM" (layers/class-name (:lstm built))))
    (testing "the named constructor fns produce the same layers"
      (is (= "Conv2D" (layers/class-name (layers/conv2d {:filters 1 :kernel-size 1}))))
      (is (= "Embedding" (layers/class-name (layers/embedding {:input-dim 2 :output-dim 2}))))
      (is (= "GRU" (layers/class-name (layers/gru {:units 1}))))
      (is (= "Flatten" (layers/class-name (layers/flatten))))
      (is (= "BatchNormalization" (layers/class-name (layers/batch-normalization))))
      (is (= "MaxPooling2D" (layers/class-name (layers/max-pooling2d {:pool-size [2 2]}))))
      (is (= "Bidirectional" (layers/class-name (layers/bidirectional {:layer (layers/gru {:units 1})})))))
    (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(layers/layer :bogus {})))))))

(deftest apply-eager-builds-weights
  (let [l (layers/dense {:units 2 :input-shape [3]})
        out (layers/apply-layer l (tf/ones [1 3]))]
    (is (tf/tensor? out))
    (is (= [1 2] (tf/shape out)))
    (is (layers/built? l))
    (is (= 8 (layers/count-params l)) "3*2 kernel + 2 bias")
    (is (= [[3 2] [2]] (mapv tf/shape (layers/weights l))))
    (is (= [nil 2] (layers/output-shape (doto (layers/dense {:units 2}) (layers/apply-layer (layers/input {:shape [3]})))))
        "output-shape is defined once the layer sits in a graph")
    (is (= :attribute-error (util/error-code (thrown #(layers/output-shape l)))) "eager-only layers have no output shape (AttributeError)")
    (is (= [nil 3] (layers/batch-input-shape l)))
    (is (= [7 2] (layers/compute-output-shape l [7 3])))
    (is (= {:num-disposed-variables 2 :ref-count-after-dispose 0} (layers/dispose! l)))))

(deftest set-weights-and-trainable
  (let [l (layers/dense {:units 1 :input-shape [2]})]
    (layers/build! l [nil 2])
    (layers/set-weights! l [(tf/tensor [[2] [3]]) (tf/tensor [1])])
    (is (= [[6]] (tf/->vec (layers/apply-layer l (tf/tensor [[1 1]])))) "1*2 + 1*3 + 1")
    (is (layers/trainable? l))
    (layers/set-trainable! l false)
    (is (not (layers/trainable? l)))
    (is (= 0 (count (layers/weights l true))) "no trainable weights once frozen")
    (is (= 2 (count (layers/weights l))))))

(deftest symbolic-inputs-and-graphs
  (let [a (layers/input {:shape [4] :name "a"})
        b (layers/input {:shape [4]})
        h (layers/apply-layer (layers/dense {:units 2}) a)
        m (layers/apply-layer (layers/concatenate) [a b])]
    (is (layers/symbolic? a))
    (is (not (layers/symbolic? (tf/ones [1]))))
    (is (= [nil 4] (layers/symbolic-shape a)))
    (is (= "a" (layers/symbolic-name a)))
    (is (= [nil 2] (layers/symbolic-shape h)))
    (is (= [nil 8] (layers/symbolic-shape m)) "merge layers take a vector of inputs")
    (is (= [nil 4] (layers/symbolic-shape (layers/apply-layer (layers/add) [a b]))))
    (is (= [nil 4] (layers/symbolic-shape (layers/apply-layer (layers/dropout {:rate 0.5}) a {:training true}))))
    (let [e (thrown #(layers/apply-layer (layers/concatenate) a))]
      (is (= :value-error (util/error-code e)) "tfjs-layers ValueError classified by constructor name")
      (is (= "ValueError" (:name (util/error-data e)))))
    (is (= :value-error (util/error-code (thrown #(layers/apply-layer (layers/add) [a (layers/input {:shape [3]})])))))))

(deftest initializer-regularizer-constraint-objects
  (let [l (layers/dense {:units 1 :use-bias false :kernel-initializer (layers/initializer :constant {:value 0.5})})]
    (is (= [[1]] (tf/->vec (layers/apply-layer l (tf/ones [1 2]))))))
  (let [l (layers/dense {:units 1 :use-bias false :kernel-initializer {:type :constant :value 2}})]
    (is (= [[4]] (tf/->vec (layers/apply-layer l (tf/ones [1 2])))) "map form {:type ..} is constructed"))
  (let [l (layers/dense {:units 2 :kernel-initializer :zeros :bias-initializer :ones})]
    (is (= [[1 1]] (tf/->vec (layers/apply-layer l (tf/ones [1 2]))))))
  (let [l (layers/dense {:units 2 :kernel-initializer :glorot-uniform :kernel-regularizer (layers/regularizer :l2 {:l2 0.01})
                         :kernel-constraint (layers/constraint :max-norm {:max-value 1})
                         :bias-regularizer {:type :l1l2 :l1 0.01 :l2 0.01} :bias-constraint {:type :non-neg}})]
    (is (= [1 2] (tf/shape (layers/apply-layer l (tf/ones [1 2]))))))
  (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(layers/dense {:units 1 :kernel-initializer :bogus})))))
  (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(layers/initializer :bogus)))))
  (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(layers/regularizer :l3)))))
  (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(layers/constraint :bogus)))))
  (is (= 15 (count layers/initializers)))
  (is (= 4 (count layers/constraints))))

(deftest activation-registry
  (is (= "gelu_new" (:activation (layers/config (layers/activation {:activation :gelu-new})))) "the one snake_case identifier")
  (is (= "gelu_new" (util/enum-value :gelu-new)))
  (is (= 15 (count layers/activations)))
  (doseq [a (keys layers/activations)]
    (is (layers/layer? (layers/activation {:activation a})) (str "activation " a)))
  (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(layers/dense {:units 1 :activation :bogus}))))
      "unknown activations are rejected by the registry before tfjs")
  (is (= "hardSigmoid" (:recurrent-activation (layers/config (layers/lstm {:units 1 :recurrent-activation :hard-sigmoid})))))
  (is (approx= [[0 2]] (tf/->vec (layers/apply-layer (layers/activation {:activation :relu}) (tf/tensor [[-1 2]]))))))

(deftest conv-and-rnn-config-render
  (let [c (layers/config (layers/conv2d {:filters 4 :kernel-size [3 3] :padding :same :data-format :channels-last
                                         :strides [1 1] :dilation-rate 1 :activation :relu}))]
    (is (= 4 (:filters c)))
    (is (= [3 3] (:kernel-size c)))
    (is (= "same" (:padding c)))
    (is (= "channelsLast" (:data-format c)))
    (is (= [1 1] (:strides c))))
  (let [c (layers/config (layers/lstm {:units 2 :return-sequences true :dropout 0.1 :unit-forget-bias true}))]
    (is (= 2 (:units c)))
    (is (true? (:return-sequences c)))
    (is (= 0.1 (:dropout c))))
  (let [l (layers/lstm {:units 3 :input-shape [5 2] :return-sequences true})
        out (layers/apply-layer l (tf/zeros [1 5 2]))]
    (is (= [1 5 3] (tf/shape out))))
  (let [l (layers/conv2d {:filters 2 :kernel-size 2 :input-shape [4 4 1]})]
    (is (= [1 3 3 2] (tf/shape (layers/apply-layer l (tf/zeros [1 4 4 1]))))))
  (let [l (layers/max-pooling2d {:pool-size [2 2]})]
    (is (= [[[[4]]]] (tf/->vec (layers/apply-layer l (tf/tensor [[[[1] [2]] [[3] [4]]]]))))))
  (let [l (layers/embedding {:input-dim 3 :output-dim 2 :embeddings-initializer :ones})]
    (is (= [[[1 1] [1 1]]] (tf/->vec (layers/apply-layer l (tf/tensor [[0 2]] :int32)))))))

(deftest generic-layer-config-passthrough
  (let [w [(tf/tensor [[1] [1]]) (tf/tensor [0])]
        l (layers/dense {:units 1 :input-shape [2] :weights w})]
    (is (= [[2]] (tf/->vec (layers/apply-layer l (tf/tensor [[1 1]])))) ":weights passthrough seeds the layer"))
  (let [l (layers/reshape {:target-shape [2 2] :input-shape [4]})]
    (is (= [1 2 2] (tf/shape (layers/apply-layer l (tf/ones [1 4]))))))
  (let [l (layers/flatten {:input-shape [2 2]})]
    (is (= [1 4] (tf/shape (layers/apply-layer l (tf/ones [1 2 2]))))))
  (let [l (layers/simple-rnn {:units 2 :stateful true :batch-input-shape [1 3 1]})]
    (layers/apply-layer l (tf/ones [1 3 1]))
    (is (identical? l (layers/reset-states! l)))))

(deftest pooling-and-global-layers
  (let [img (tf/tensor [[[[1] [2]] [[3] [4]]]])]
    (is (= [[[[2.5]]]] (tf/->vec (layers/apply-layer (layers/average-pooling2d {:pool-size [2 2]}) img))))
    (is (= [[[[4]]]] (tf/->vec (layers/apply-layer (layers/layer :max-pool2d {:pool-size 2}) img))) "alias constructor")
    (is (= [[4]] (tf/->vec (layers/apply-layer (layers/global-max-pooling2d) img))))
    (is (= [[2.5]] (tf/->vec (layers/apply-layer (layers/global-average-pooling2d) img))))
    (is (= [1 4 4 1] (tf/shape (layers/apply-layer (layers/up-sampling2d {:size [2 2]}) img))))
    (is (= [1 4 4 1] (tf/shape (layers/apply-layer (layers/zero-padding2d {:padding 1}) img))))
    (is (= [1 1 1 1] (tf/shape (layers/apply-layer (layers/cropping2d {:cropping [[1 0] [0 1]]}) img)))))
  (let [seq3 (tf/tensor [[[1] [2] [3]]])]
    (is (= [[2]] (tf/->vec (layers/apply-layer (layers/global-average-pooling1d) seq3))))
    (is (= [[3]] (tf/->vec (layers/apply-layer (layers/global-max-pooling1d) seq3))))
    (is (= [1 1 1] (tf/shape (layers/apply-layer (layers/max-pooling1d {:pool-size 3}) seq3))))
    (is (= [1 2 1] (tf/shape (layers/apply-layer (layers/average-pooling1d {:pool-size 2 :strides 1}) seq3))))))

(deftest wrapper-and-recurrent-layers
  (let [x (tf/zeros [1 4 3])]
    (is (= [1 4] (tf/shape (layers/apply-layer (layers/bidirectional {:layer (layers/gru {:units 2})}) x))) "forward + backward concatenated")
    (is (= [1 2] (tf/shape (layers/apply-layer (layers/bidirectional {:layer (layers/gru {:units 2}) :merge-mode :sum}) x))))
    (is (= [1 4 5] (tf/shape (layers/apply-layer (layers/time-distributed {:layer (layers/dense {:units 5})}) x))))
    (is (= [1 4 2] (tf/shape (layers/apply-layer (layers/simple-rnn {:units 2 :return-sequences true}) x))))
    (is (= [1 2] (tf/shape (layers/apply-layer (layers/rnn {:cell (layers/lstm-cell {:units 2})}) x))))
    (is (= [1 3] (tf/shape (layers/apply-layer (layers/rnn {:cell (layers/stacked-rnn-cells {:cells [(layers/gru-cell {:units 2}) (layers/gru-cell {:units 3})]})}) x))))
    (let [[out h c] (layers/apply-layer (layers/lstm {:units 2 :return-state true}) x)]
      (is (= [1 2] (tf/shape out)) "return-state gives a vector [output h c]")
      (is (= [1 2] (tf/shape h)))
      (is (= [1 2] (tf/shape c))))
    (is (= [1 3 3] (tf/shape (layers/apply-layer (layers/repeat-vector {:n 3}) (tf/zeros [1 3])))))
    (is (= [1 3 4] (tf/shape (layers/apply-layer (layers/permute {:dims [2 1]}) x))))
    (is (= [1 4 3] (tf/shape (layers/apply-layer (layers/masking {:mask-value 0}) x))))))

(deftest normalization-noise-and-preprocessing-layers
  (let [x (tf/tensor [[1 2] [3 4]])]
    (is (= [2 2] (tf/shape (layers/apply-layer (layers/batch-normalization) x))))
    (is (approx= [[-1 1] [-1 1]] (tf/->vec (layers/apply-layer (layers/layer-normalization {:epsilon 1e-7}) x)) 1e-3))
    (is (= [2 2] (tf/shape (layers/apply-layer (layers/gaussian-noise {:stddev 0.1}) x {:training true}))))
    (is (= [2 2] (tf/shape (layers/apply-layer (layers/gaussian-dropout {:rate 0.1}) x {:training true}))))
    (is (= [2 2] (tf/shape (layers/apply-layer (layers/alpha-dropout {:rate 0.1}) x {:training true}))))
    (is (= [[3 5] [7 9]] (tf/->vec (layers/apply-layer (layers/rescaling {:scale 2 :offset 1}) x))))
    (is (= [[0 1 0] [0 0 1]] (tf/->vec (layers/apply-layer (layers/category-encoding {:num-tokens 3 :output-mode :one-hot}) (tf/tensor [1 2])))))
    (is (= [[1 0 1]] (tf/->vec (layers/apply-layer (layers/category-encoding {:num-tokens 3 :output-mode :multi-hot}) (tf/tensor [[0 2]])))))
    (is (= [1 2 2 1] (tf/shape (layers/apply-layer (layers/resizing {:height 2 :width 2}) (tf/ones [1 4 4 1])))))
    (is (= [1 1 1 1] (tf/shape (layers/apply-layer (layers/center-crop {:height 1 :width 1}) (tf/ones [1 4 4 1])))))
    (is (approx= [[0 2]] (tf/->vec (layers/apply-layer (layers/leaky-relu {:alpha 0}) (tf/tensor [[-1 2]])))))
    (is (approx= [[0 0 2]] (tf/->vec (layers/apply-layer (layers/thresholded-relu {:theta 1}) (tf/tensor [[-1 1 2]])))))
    (is (= [[0 6]] (tf/->vec (layers/apply-layer (layers/relu {:max-value 6}) (tf/tensor [[-1 10]])))))
    (is (approx= [[0.5 0.5]] (tf/->vec (layers/apply-layer (layers/softmax) (tf/tensor [[1 1]])))))
    (is (= [1 2] (tf/shape (layers/apply-layer (layers/prelu) (tf/tensor [[-1 2]])))))
    (is (= [1 2] (tf/shape (layers/apply-layer (layers/elu {:alpha 1}) (tf/tensor [[-1 2]])))))))
