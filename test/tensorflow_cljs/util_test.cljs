(ns tensorflow-cljs.util-test
  (:require [cljs.test :refer [deftest is testing]]
            [clojure.string :as str]
            [tensorflow-cljs.util :as util]))

(deftest kebab-camel-roundtrip
  (is (= "kernelInitializer" (util/kebab->camel :kernel-initializer)))
  (is (= "batchInputShape" (util/kebab->camel :batch-input-shape)))
  (is (= "units" (util/kebab->camel :units)) "single word unchanged")
  (is (= "alreadyCamel" (util/kebab->camel "alreadyCamel")) "strings pass through")
  (is (= "validationSplit" (util/kebab->camel :tensorflow-cljs/validation-split)) "namespace dropped")
  (is (= :kernel-initializer (util/camel->kebab "kernelInitializer")))
  (is (= :batch-input-shape (util/camel->kebab :batchInputShape)))
  (is (= :val-loss (util/camel->kebab "val_loss")) "snake_case history keys")
  (is (= :units (util/camel->kebab "units")))
  (is (= :tensorflow-cljs/x (util/camel->kebab :tensorflow-cljs/x)) "namespaced keywords untouched")
  (doseq [k [:units :activation :kernel-initializer :bias-initializer :input-shape
             :batch-input-shape :validation-split :batch-size :epochs :shuffle
             :return-sequences :data-format :pool-size :kernel-size :use-bias
             :yield-every :learning-rate]]
    (is (= k (util/camel->kebab (util/kebab->camel k))) (str "roundtrip " k))))

(deftest transform-keys-honours-passthrough
  (is (= {"aB" {"cD" 1} "metadata" {:x-y 2}}
         (util/transform-keys util/kebab->camel #{:metadata} {:a-b {:c-d 1} :metadata {:x-y 2}})))
  (is (= [{:a-b 1}] (util/transform-keys util/camel->kebab [{:aB 1}]))))

(deftest enum-values-render-as-registry-names
  (is (= "relu" (util/enum-value :relu)))
  (is (= "glorotUniform" (util/enum-value :glorot-uniform)))
  (is (= "categoricalCrossentropy" (util/enum-value :categorical-crossentropy)))
  (is (= "channelsLast" (util/enum-value :channels-last)))
  (is (= "same" (util/enum-value :same)))
  (is (= "float32" (util/enum-value :float32)))
  (is (= 3 (util/enum-value 3)))
  (is (= "meanSquaredError" (util/enum-value "meanSquaredError")) "strings untouched"))

(deftest ->config-drops-nil-and-keeps-user-data
  (let [f (fn [])
        o (util/->config {:units 4 :activation :relu :kernel-initializer :glorot-uniform
                          :input-shape [2 3] :use-bias nil
                          :metadata {:user-key 1}
                          :callbacks {:on-epoch-end f}
                          :nested {:learning-rate 0.01 :name :adam}})]
    (is (= {"units" 4 "activation" "relu" "kernelInitializer" "glorotUniform"
            "inputShape" [2 3]
            "metadata" {:user-key 1}
            "callbacks" {:on-epoch-end f}
            "nested" {"learningRate" 0.01 "name" "adam"}}
           o)
        "keys camelised, nil dropped, enums rendered, shapes numeric, passthrough subtrees kept")
    (is (= {"batchSize" 2} (util/->config {"batchSize" 2})) "string keys pass")
    (is (nil? (util/->config nil)))))

(deftest ->kebab-keys-translates-history-and-memory
  (is (= {:num-tensors 3 :num-data-buffers 3 :num-bytes 48 :unreliable true}
         (util/->kebab-keys {"numTensors" 3 "numDataBuffers" 3 "numBytes" 48 "unreliable" true})))
  (is (= {:history {:loss [1 2] :val-loss [3 4] :val-acc [0.5]}}
         (util/->kebab-keys {"history" {"loss" [1 2] "val_loss" [3 4] "val_acc" [0.5]}}))))

(deftest dtypes-and-shapes
  (is (= "float32" (util/dtype->str :float32)))
  (is (= "int32" (util/dtype->str "int32")))
  (is (= "complex64" (util/dtype->str :complex64)))
  (is (nil? (util/dtype->str nil)))
  (let [e (try (util/dtype->str :float64) nil (catch :default e e))]
    (is (some? e))
    (is (= :tensorflow-cljs/argument-error (util/error-type e)))
    (is (= #{:float32 :int32 :bool :string :complex64} (:allowed (ex-data e)))))
  (is (= :float32 (util/str->dtype "float32")))
  (is (= :int32 (util/str->dtype :int32)))
  (is (util/shape? [2 3]))
  (is (util/shape? []))
  (is (not (util/shape? [2 -1])))
  (is (not (util/shape? [2 1.5])))
  (is (not (util/shape? 3)))
  (is (= 6 (util/size-of-shape [2 3])))
  (is (= 1 (util/size-of-shape [])))
  (is (= [2 3] (util/infer-shape [[1 2 3] [4 5 6]])))
  (is (= [2 1 2] (util/infer-shape [[[1 2]] [[3 4]]])))
  (is (= [] (util/infer-shape 3)))
  (is (not (util/ragged? [[1 2] [3 4]])))
  (is (not (util/ragged? [1 2 3])))
  (is (not (util/ragged? 3)))
  (is (not (util/ragged? [])))
  (is (util/ragged? [1 [2 3]]) "scalar first, array later")
  (is (util/ragged? [[1 2] [3]]))
  (is (util/ragged? [[1 2] [3 4 5] [6]]) "count-compatible but ragged")
  (is (util/ragged? [[1] 2]))
  (is (util/ragged? [[[1 2]] [[3]]])))

(deftest tf-error-classification
  (is (= :value-error (util/classify-tf-error "ValueError" "Input 0 is incompatible")))
  (is (= :runtime-error (util/classify-tf-error "RuntimeError" "x")))
  (is (= :not-implemented-error (util/classify-tf-error "NotImplementedError" "x")))
  (is (= :attribute-error (util/classify-tf-error "AttributeError" "x")))
  (is (= :index-error (util/classify-tf-error "IndexError" "x")))
  (is (= :shape-error (util/classify-tf-error "Error" "Error in matMul: inner shapes (2) and (3) of Tensors with shapes 2,2 and 3,2 and transposeA=false and transposeB=false must match.")))
  (is (= :shape-error (util/classify-tf-error "Error" "Operands could not be broadcast together with shapes 2,3 and 4.")))
  (is (= :shape-error (util/classify-tf-error "Error" "Argument 'indices' passed to 'oneHot' must be int32 tensor, but got float32 tensor")))
  (is (= :shape-error (util/classify-tf-error "Error" "Argument 'a' passed to 'add' must be numeric tensor, but got string tensor")))
  (is (= :assertion-error (util/classify-tf-error "Error" "Error in tensor2d: values must be a flat/TypedArray")))
  (is (= :unknown (util/classify-tf-error "Error" "something else")))
  (is (= :unknown (util/classify-tf-error nil nil))))

(deftest error-constructors-and-detection
  (let [tfe (util/tf-error "ValueError" "Input 0 is incompatible with layer dense_Dense1" {:cause :js-err})
        shp (util/tf-error "Error" "Error in matMul: inner shapes (2) and (3) ... must match.")
        arg (util/argument-error "bad" {:arg :shape})
        io (util/io-error "model.json not found" {:path "file://x"})
        be (util/backend-error "webgl unavailable" {:backend :webgl})]
    (is (= :tensorflow-cljs/tf-error (util/error-type tfe)))
    (is (= :value-error (util/error-code tfe)))
    (is (= "ValueError" (:name (util/error-data tfe))))
    (is (= :js-err (:cause (util/error-data tfe))))
    (is (= "tensorflow-cljs value-error: Input 0 is incompatible with layer dense_Dense1" (ex-message tfe)))
    (is (not (util/shape-error? tfe)))
    (is (util/shape-error? shp))
    (is (= :shape-error (util/error-code shp)))
    (is (= :tensorflow-cljs/argument-error (util/error-type arg)))
    (is (= :shape (:arg (ex-data arg))))
    (is (nil? (util/error-code arg)) "only tf errors carry codes")
    (is (= :tensorflow-cljs/io-error (util/error-type io)))
    (is (str/starts-with? (ex-message io) "tensorflow-cljs io:"))
    (is (= :tensorflow-cljs/backend-error (util/error-type be)))
    (is (= :webgl (:backend (ex-data be))))
    (is (nil? (util/error-data (ex-info "other" {:type :something/else}))))
    (is (nil? (util/error-data (js/Error. "plain"))))))
