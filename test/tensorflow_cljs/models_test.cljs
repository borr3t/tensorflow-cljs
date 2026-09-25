(ns tensorflow-cljs.models-test
  "Sequential / functional models: compile, fit, predict, evaluate,
   callbacks, save / load round trips, tf.data training."
  (:require [cljs.test :refer [deftest is testing async]]
            [clojure.string :as str]
            [promesa.core :as p]
            [tensorflow-cljs.core :as tf]
            [tensorflow-cljs.data :as data]
            [tensorflow-cljs.layers :as layers]
            [tensorflow-cljs.models :as models]
            [tensorflow-cljs.test-util :as tu :refer [approx= thrown]]
            [tensorflow-cljs.util :as util]))

(defn- tiny-model []
  (models/sequential [(layers/dense {:units 3 :input-shape [2] :activation :relu})
                      (layers/dense {:units 1})]))

(def xor-x (tf/tensor [[0 0] [0 1] [1 0] [1 1]]))
(def xor-y (tf/tensor [[0] [1] [1] [0]]))

(deftest sequential-predict-round-trip
  (let [m (tiny-model)
        out (models/predict m (tf/tensor [[1 2]]))]
    (is (models/model? m))
    (is (models/sequential? m))
    (is (tf/tensor? out))
    (is (= [1 1] (tf/shape out)))
    (is (= :float32 (tf/dtype (models/predict m (tf/tensor [[1 2]] :int32)))) "predict outputs float32")
    (is (= [2 1] (tf/shape (models/predict m (tf/tensor [[1 2] [3 4]]) {:batch-size 1}))))
    (is (= [1 1] (tf/shape (models/predict-on-batch m (tf/tensor [[1 2]])))))
    (is (= 13 (models/count-params m)) "2*3+3 + 3*1+1")
    (is (= 2 (count (models/layers m))))
    (is (string? (models/model-name m)))
    (is (not (models/compiled? m)))
    (is (nil? (models/metrics-names m)))
    (is (= :value-error (util/error-code (thrown #(models/predict m (tf/tensor [[1 2 3]]))))) "shape mismatch is a ValueError")
    (models/add-layer! m (layers/dense {:units 4}))
    (is (= 3 (count (models/layers m))))
    (models/pop-layer! m)
    (is (= 2 (count (models/layers m))))
    (let [empty-m (models/sequential)]
      (models/add-layer! empty-m (layers/dense {:units 1 :input-shape [1]}))
      (is (= 1 (count (models/layers empty-m)))))
    (is (= :value-error (util/error-code (thrown #(models/sequential [(layers/dense {:units 1})]))))
        "first layer without input-shape")))

(deftest compile-translates-registries
  (let [m (tiny-model)]
    (is (identical? m (models/compile-model! m {:optimizer :adam :loss :mean-squared-error :metrics [:mae :accuracy]})))
    (is (models/compiled? m))
    (is (= ["loss" "mae" "acc"] (models/metrics-names m)))
    (models/compile! m {:optimizer :sgd :loss :mse :metrics [:mse]})
    (is (= ["loss" "mse"] (models/metrics-names m)) ":mse alias and recompilation")
    (models/compile-model! m {:optimizer {:type :sgd :learning-rate 0.5} :loss (fn [y-true y-pred] (tf/mean (tf/abs (tf/sub y-true y-pred))))})
    (is (= ["loss"] (models/metrics-names m)) "optimizer map form + custom loss fn")
    (models/compile-model! m {:optimizer (models/optimizer :rmsprop {:learning-rate 0.01}) :loss :binary-crossentropy :metrics {"dense_Dense2" [:accuracy]}})
    (is (tf/optimizer? (models/optimizer :adam {:learning-rate 0.5})))
    (is (= "Adam" (tf/optimizer-class-name (models/optimizer :adam))))
    (is (= "Momentum" (tf/optimizer-class-name (models/optimizer :momentum {:learning-rate 0.1 :momentum 0.9}))))
    (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(models/compile-model! m {:optimizer :adam :loss :bogus})))))
    (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(models/compile-model! m {:optimizer :bogus :loss :mse})))))
    (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(models/compile-model! m {:optimizer :adam :loss :mse :metrics [:bogus]})))))
    (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(models/compile-model! m {:optimizer :adam})))))
    (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(models/optimizer :bogus)))))
    (is (= :value-error (util/error-code (thrown #(models/compile-model! m {:optimizer "bogus" :loss :mse}))))
        "unknown strings reach tfjs and come back as ValueError")
    (is (= 14 (count (dissoc models/losses :mse :mae :mape :msle :kld :cosine))) "14 compile losses")))

(deftest xor-trains-to-100-percent
  (async done
    (tu/finish
     (let [m (models/sequential [(layers/dense {:units 8 :input-shape [2] :activation :tanh :kernel-initializer {:type :glorot-uniform :seed 7}})
                                 (layers/dense {:units 1 :activation :sigmoid})])]
       (models/compile-model! m {:optimizer {:type :adam :learning-rate 0.1} :loss :binary-crossentropy :metrics [:accuracy]})
       (p/let [h (models/fit! m xor-x xor-y {:epochs 150 :batch-size 4 :yield-every :never})
               ev (models/evaluate! m xor-x xor-y)
               pred (models/predict! m xor-x)]
         (is (= 150 (count (:epoch h))))
         (is (= 150 (count (get-in h [:history :loss]))))
         (is (< (last (get-in h [:history :loss])) (first (get-in h [:history :loss]))))
         (is (= 1 (:acc ev)) (str "XOR accuracy 100%; final loss " (last (get-in h [:history :loss]))))
         (is (= [0 1 1 0] (mapv #(js/Math.round (first %)) (tf/->vec pred))))))
     done)))

(deftest linear-regression-converges
  (async done
    (tu/finish
     (let [xs (tf/tensor (mapv vector (range 10)))
           ys (tf/tensor (mapv (fn [x] [(+ 1 (* 2 x))]) (range 10)))
           m (models/sequential [(layers/dense {:units 1 :input-shape [1]})])]
       (models/compile-model! m {:optimizer {:type :sgd :learning-rate 0.01} :loss :mean-squared-error})
       (p/let [h (models/fit! m xs ys {:epochs 40 :batch-size 10 :shuffle false :yield-every :never})
               losses (get-in h [:history :loss])]
         (is (= 40 (count losses)))
         (is (< (last losses) (* 0.1 (first losses))) (str "loss decreases: " (first losses) " -> " (last losses)))
         (is (= 40 (get-in h [:params :epochs])) "params kebab-cased")
         (is (= 10 (get-in h [:params :batch-size])))))
     done)))

(deftest fit-history-with-validation
  (async done
    (tu/finish
     (let [m (models/compile-model! (tiny-model) {:optimizer :sgd :loss :mse :metrics [:mae]})]
       (p/let [h (models/fit! m xor-x xor-y {:epochs 2 :validation-split 0.5 :batch-size 2 :yield-every :never})]
         (is (= [0 1] (:epoch h)))
         (is (= #{:loss :mae :val-loss :val-mae} (set (keys (:history h)))) "snake_case history keys kebab-cased")
         (is (every? number? (get-in h [:history :val-loss])))
         (is (true? (get-in h [:params :do-validation])))
         (is (= 2 (count (get-in h [:history :val-mae]))))))
     done)))

(deftest callbacks-are-invoked
  (async done
    (tu/finish
     (let [m (models/compile-model! (tiny-model) {:optimizer :sgd :loss :mse})
           events (atom [])
           rec (fn [k] (fn [& args] (swap! events conj (into [k] args)) nil))]
       (p/let [_ (models/fit! m xor-x xor-y {:epochs 2 :batch-size 2 :yield-every :never
                                             :callbacks {:on-train-begin (rec :train-begin)
                                                         :on-train-end (rec :train-end)
                                                         :on-epoch-begin (rec :epoch-begin)
                                                         :on-epoch-end (fn [epoch logs] (swap! events conj [:epoch-end epoch logs]) (p/resolved nil))
                                                         :on-batch-begin (rec :batch-begin)
                                                         :on-batch-end (rec :batch-end)}})
               evs @events
               by-kind (group-by first evs)]
         (is (= 1 (count (:train-begin by-kind))))
         (is (= 1 (count (:train-end by-kind))))
         (is (= 2 (count (:epoch-begin by-kind))))
         (is (= 2 (count (:epoch-end by-kind))))
         (is (= 4 (count (:batch-end by-kind))) "2 epochs x 2 batches")
         (is (= [0 1] (mapv second (:epoch-end by-kind))))
         (is (number? (:loss (nth (first (:epoch-end by-kind)) 2))) "logs arrive as kebab maps of numbers")
         (is (contains? (nth (first (:batch-end by-kind)) 2) :batch))
         (is (= :train-begin (first (first evs))))
         (is (= :train-end (first (last evs))))))
     done)))

(deftest early-stopping-and-stop-training
  (async done
    (tu/finish
     (let [m (models/compile-model! (tiny-model) {:optimizer :sgd :loss :mse})
           epochs (atom 0)]
       (p/let [h (models/fit! m xor-x xor-y {:epochs 5 :batch-size 4 :yield-every :never
                                             :callbacks [(models/early-stopping {:monitor :loss :patience 0 :min-delta 1e9})
                                                         {:on-epoch-end (fn [_ _] (swap! epochs inc))}
                                                         (models/callback {:on-epoch-begin (fn [_ _] nil)})]})]
         (is (<= (count (:epoch h)) 2) "early stopping with an impossible min-delta halts after the first check")
         (is (= (count (:epoch h)) @epochs) "plain hook maps mix with callback objects"))
       (p/let [seen (atom 0)
               h (models/fit! m xor-x xor-y {:epochs 5 :batch-size 4 :yield-every :never
                                             :callbacks {:on-epoch-end (fn [epoch _] (swap! seen inc) (when (= 1 epoch) (models/stop-training! m)))}})]
         (is (= 2 (count (:epoch h))) "stop-training! ends after the current epoch")
         (is (= 2 @seen)))
       (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(models/callback {:on-bogus (fn [])}))))))
     done)))

(deftest evaluate-returns-metric-map
  (let [m (models/compile-model! (tiny-model) {:optimizer :sgd :loss :mse :metrics [:mae]})
        before (tf/num-tensors)
        ev (models/evaluate m xor-x xor-y)]
    (is (= #{:loss :mae} (set (keys ev))))
    (is (every? number? (vals ev)))
    (is (= before (tf/num-tensors)) "evaluate scalars are read and disposed (no leak)")
    (let [b2 (tf/num-tensors)
          out (models/predict m xor-x)]
      (is (= (inc b2) (tf/num-tensors)) "predict leaks only its output")
      (tf/dispose! out))
    (async done
      (tu/finish
       (p/let [ev2 (models/evaluate! m xor-x xor-y {:batch-size 2})
               tob (models/train-on-batch! m xor-x xor-y)
               e (tu/rejection (models/fit! (tiny-model) xor-x xor-y))]
         (is (= (set (keys ev)) (set (keys ev2))))
         (is (vector? tob) "train-on-batch! with metrics -> [loss mae]")
         (is (= 2 (count tob)))
         (is (= :runtime-error (util/error-code e)) "fit before compile rejects with RuntimeError"))
       done))))

(deftest save-and-load-round-trip-via-temp-dir
  (async done
    (let [dir (tu/temp-dir! "tfcljs-")]
      (tu/finish
       (let [m (models/compile-model! (tiny-model) {:optimizer :adam :loss :mse :metrics [:mae]})
             x (tf/tensor [[0.5 -1]])
             expected (tf/->vec (models/predict m x))]
         (models/set-user-metadata! m {"trained-by" "tensorflow-cljs"})
         (p/let [saved (models/save! m dir)
                 m2 (models/load! dir)
                 saved-with-opt (models/save! m (tu/join-path dir "opt") {:include-optimizer true})
                 m3 (models/load! (tu/join-path dir "opt" "model.json"))
                 m4 (models/load! (str "file://" dir))]
           (is (= dir (:dir saved)))
           (is (tu/file-exists? (tu/join-path dir "model.json")))
           (is (tu/file-exists? (tu/join-path dir "weights.bin")))
           (is (= "JSON" (get-in saved [:model-artifacts-info :model-topology-type])))
           (is (= 52 (get-in saved [:model-artifacts-info :weight-data-bytes])) "13 float32 params")
           (is (re-find #"weightsManifest" (tu/read-file (tu/join-path dir "model.json"))))
           (is (models/model? m2))
           (is (approx= expected (tf/->vec (models/predict m2 x))) "weights survive the round trip")
           (is (not (models/compiled? m2)) "without :include-optimizer the training config is not stored")
           (is (= {"trained-by" "tensorflow-cljs"} (models/user-metadata m2)))
           (is (models/compiled? m3) ":include-optimizer stores the training config")
           (is (= ["loss" "mae"] (models/metrics-names m3)))
           (is (approx= expected (tf/->vec (models/predict m3 x))))
           (is (approx= expected (tf/->vec (models/predict m4 x))) "file:// URL accepted")
           (is (> (get-in saved-with-opt [:model-artifacts-info :weight-data-bytes]) 52) "optimizer state adds weights")
           (p/let [h (models/fit! m3 xor-x xor-y {:epochs 1 :yield-every :never})]
             (is (= 1 (count (:epoch h))) "a loaded compiled model trains without compile!"))))
       (fn [] (tu/rm-rf! dir) (done))))))

(deftest load-concatenates-multi-shard-manifests
  (async done
    (let [dir (tu/temp-dir! "tfcljs-shards-")]
      (tu/finish
       (let [m (tiny-model)
             x (tf/tensor [[1 2]])
             expected (tf/->vec (models/predict m x))]
         (p/let [_ (models/save! m dir)
                 _ (do (tu/split-binary-file! (tu/join-path dir "weights.bin") (tu/join-path dir "a.bin") (tu/join-path dir "b.bin") 20)
                       (tu/write-file! (tu/join-path dir "model.json")
                                       (str/replace (tu/read-file (tu/join-path dir "model.json"))
                                                    "[\"./weights.bin\"]" "[\"a.bin\",\"b.bin\"]")))
                 m2 (models/load! dir)
                 missing (tu/rejection (models/load! (tu/join-path dir "nope")))]
           (is (approx= expected (tf/->vec (models/predict m2 x))) "two shards concatenated in manifest order")
           (is (= :tensorflow-cljs/io-error (util/error-type missing)))
           (is (re-find #"model.json not found" (ex-message missing)))
           (tu/write-file! (tu/join-path dir "model.json")
                           (str/replace (tu/read-file (tu/join-path dir "model.json")) "b.bin" "gone.bin"))
           (p/let [e (tu/rejection (models/load! dir))]
             (is (= :tensorflow-cljs/io-error (util/error-type e)))
             (is (re-find #"gone.bin" (ex-message e))))))
       (fn [] (tu/rm-rf! dir) (done))))))

(deftest in-memory-artifacts-and-json
  (async done
    (tu/finish
     (let [m (tiny-model)
           x (tf/tensor [[3 4]])
           expected (tf/->vec (models/predict m x))]
       (p/let [artifacts (models/save-artifacts! m)
               m2 (models/load-artifacts! artifacts)
               info (models/artifacts-info artifacts)
               specs (models/weight-specs artifacts)
               topo (models/to-json m)
               m3 (models/model-from-json! topo)]
         (is (approx= expected (tf/->vec (models/predict m2 x))))
         (is (= 52 (:weight-data-bytes info)))
         (is (= 4 (count specs)))
         (is (= [3] (:shape (second specs))))
         (is (str/ends-with? (:name (first specs)) "/kernel"))
         (is (= "float32" (:dtype (first specs))))
         (is (= "Sequential" (get topo "class_name")) "topology keys verbatim")
         (is (= 2 (count (models/layers m3))))
         (is (= [1 1] (tf/shape (models/predict m3 x))))))
     done)))

(deftest functional-model-with-two-inputs
  (let [a (layers/input {:shape [2] :name "a"})
        b (layers/input {:shape [2] :name "b"})
        merged (layers/apply-layer (layers/concatenate) [a b])
        out (layers/apply-layer (layers/dense {:units 1 :name "out"}) merged)
        m (models/model {:inputs [a b] :outputs out :name "two-in"})]
    (is (models/model? m))
    (is (not (models/sequential? m)))
    (is (= "two-in" (models/model-name m)))
    (is (= ["a" "b"] (models/input-names m)))
    (is (= ["out"] (models/output-names m)))
    (is (= 2 (count (models/inputs m))))
    (is (= 1 (count (models/outputs m))))
    (is (= [1 1] (tf/shape (models/predict m [(tf/ones [1 2]) (tf/ones [1 2])]))))
    (is (= :value-error (util/error-code (thrown #(models/predict m (tf/ones [1 2]))))) "multi-input model needs a vector")
    (is (= "out" (layers/layer-name (models/get-layer m "out"))))
    (is (= "a" (layers/layer-name (models/get-layer m 0))) "by index")
    (is (= :value-error (util/error-code (thrown #(models/get-layer m "nope")))))
    (is (= [1 4] (tf/shape (first (models/execute m [(tf/ones [1 2]) (tf/ones [1 2])] [(layers/layer-name (models/get-layer m 2))]))))
        "execute reads an intermediate layer output by layer name")
    (let [s (models/summary m)]
      (is (string? s))
      (is (re-find #"Total params: 5" s))
      (is (re-find #"concatenate" s))
      (is (= (count (str/split-lines s)) (count (models/summary-lines m)))))
    (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(models/model {:inputs a})))))
    (async done
      (tu/finish
       (p/let [_ (models/compile-model! m {:optimizer :sgd :loss :mse})
               h (models/fit! m {"a" (tf/ones [2 2]) "b" (tf/ones [2 2])} (tf/ones [2 1]) {:epochs 1 :yield-every :never})]
         (is (= 1 (count (:epoch h))) "fit accepts a map keyed by input name"))
       done))
    (is (= 4 (count (models/weights (models/model {:inputs a :outputs (layers/apply-layer (layers/dense {:units 2}) (layers/apply-layer (layers/dense {:units 2}) a))})))))))

(deftest weights-set-get-and-dispose
  (let [m (tiny-model)
        ws (models/weights m)]
    (is (= 4 (count ws)))
    (is (= [[2 3] [3] [3 1] [1]] (mapv tf/shape ws)))
    (models/set-weights! m [(tf/ones [2 3]) (tf/zeros [3]) (tf/ones [3 1]) (tf/tensor [1])])
    (is (= [[7]] (tf/->vec (models/predict m (tf/tensor [[1 1]])))) "relu(1+1)*3 + 1")
    (is (= 4 (count (models/weights m true))))
    (let [r (models/dispose! m)]
      (is (= 4 (:num-disposed-variables r)))
      (is (= 0 (:ref-count-after-dispose r))))))

(deftest fit-dataset-and-evaluate-dataset
  (async done
    (tu/finish
     (let [rows (mapv (fn [x] {:xs [x] :ys [(+ 1 (* 2 x))]}) (range 8))
           ds (data/batch (data/from-vec rows) 4)
           m (models/compile-model! (models/sequential [(layers/dense {:units 1 :input-shape [1]})])
                                    {:optimizer {:type :sgd :learning-rate 0.01} :loss :mse :metrics [:mae]})]
       (p/let [h (models/fit-dataset! m ds {:epochs 3 :yield-every :never})
               ev (models/evaluate-dataset! m ds)
               unsized (tu/rejection (models/fit-dataset! m (data/repeat ds) {:epochs 1}))
               h2 (models/fit-dataset! m (data/repeat ds) {:epochs 1 :batches-per-epoch 2 :yield-every :never})
               h3 (models/fit-dataset! m (data/filter ds (constantly true)) {:epochs 1 :batches-per-epoch 2 :yield-every :never})]
         (is (= 3 (count (get-in h [:history :loss]))))
         (is (< (last (get-in h [:history :loss])) (first (get-in h [:history :loss]))))
         (is (= #{:loss :mae} (set (keys ev))))
         (is (= :tensorflow-cljs/argument-error (util/error-type unsized)) "infinite dataset needs :batches-per-epoch")
         (is (= 1 (count (:epoch h2))))
         (is (= 1 (count (:epoch h3))) "filtered (size nil) dataset with :batches-per-epoch")))
     done)))

(deftest optimizer-constructors
  (doseq [[k spec] models/optimizer-args]
    (is (tf/optimizer? (models/optimizer k)) (str "default " k))
    (is (tf/optimizer? (models/optimizer k (into {} (map (fn [[arg d]] [arg (or d 0.5)])) spec))) (str "explicit args " k)))
  (is (= ["SGD" "Momentum" "RMSProp" "Adam" "Adadelta" "Adamax" "Adagrad"]
         (mapv #(tf/optimizer-class-name (models/optimizer %)) [:sgd :momentum :rmsprop :adam :adadelta :adamax :adagrad])))
  (is (= 7 (count models/optimizers)))
  (let [w (tf/variable (tf/scalar 1))
        opt (models/optimizer :sgd {:learning-rate 0.5})]
    (tf/minimize! opt (fn [] (tf/mul w 2)))
    (is (= 0 (tf/->vec w)) "w - 0.5 * d(2w)/dw")
    (tf/dispose! w)))

(deftest summary-and-introspection
  (let [m (models/sequential [(layers/dense {:units 3 :input-shape [2] :name "hidden"})
                              (layers/dense {:units 1 :name "head"})] {:name "tiny"})
        lines (models/summary-lines m)]
    (is (= "tiny" (models/model-name m)))
    (is (some #(re-find #"hidden \(Dense\)" %) lines))
    (is (some #(re-find #"Total params: 13" %) lines))
    (is (some #(re-find #"Trainable params: 13" %) lines))
    (is (< (count (models/summary-lines m {:line-length 60})) 20))
    (is (= ["hidden" "head"] (mapv layers/layer-name (models/layers m))))
    (is (= "head" (layers/layer-name (models/get-layer m 1))))
    (is (= 1 (count (models/input-names m))))
    (is (= [nil 2] (layers/symbolic-shape (first (models/inputs m)))))
    (is (= [nil 1] (layers/symbolic-shape (first (models/outputs m)))))
    (is (= "Sequential" (get (models/to-json m) "class_name")))
    (is (nil? (models/user-metadata m)))
    (layers/set-trainable! (models/get-layer m "hidden") false)
    (is (some #(re-find #"Non-trainable params: 9" %) (models/summary-lines m)) "freezing a layer moves its params")
    (is (= 2 (count (models/weights m true))))
    (is (string? (with-out-str (models/print-summary m))))))
