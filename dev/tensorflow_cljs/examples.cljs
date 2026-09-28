(ns tensorflow-cljs.examples
  "The examples of the README, runnable OFFLINE:

     npx shadow-cljs compile examples && node target/examples.js

   Everything is computed in this process, on the CPU backend of
   TensorFlow.js: no network, no service, no account. The one thing that
   touches the disk is the save / load example, in a directory it is given.

   What stands between a README:begin and a README:end mark stands in
   README.md as written (test/tensorflow_cljs/examples_test.cljs holds the
   README to it and runs every function of this file)."
  (:require ["fs" :as fs]
            ["os" :as os]
            ["path" :as path]
            [cljs.pprint :as pprint]
            [clojure.string :as str]
            [promesa.core :as p]
            [tensorflow-cljs.core :as tf]
            [tensorflow-cljs.data :as data]
            [tensorflow-cljs.layers :as layers]
            [tensorflow-cljs.models :as models]
            [tensorflow-cljs.util :as util]))

;; ---------------------------------------------------------------------------
;; Quick start: tensors
;; ---------------------------------------------------------------------------

;; README:begin quick-start
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
;; README:end quick-start

;; ---------------------------------------------------------------------------
;; Scenario: train a model
;; ---------------------------------------------------------------------------

;; README:begin train
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
;; README:end train

;; ---------------------------------------------------------------------------
;; Scenario: save a model, load it back
;; ---------------------------------------------------------------------------

;; README:begin save
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
;; README:end save

(defn temp-dir!
  "A fresh directory for the save / load example."
  []
  (fs/mkdtempSync (path/join (os/tmpdir) "tensorflow-cljs-examples-")))

(defn save-offline
  "`save-and-load` of a trained model, in a directory that is removed
   afterwards."
  []
  (let [model (xor-model)
        dir   (temp-dir!)]
    (-> (p/let [_ (train! model xs ys 150)]
          (save-and-load model dir))
        (p/finally (fn [_ _]
                     (models/dispose! model)
                     (fs/rmSync dir #js {:recursive true :force true}))))))

;; ---------------------------------------------------------------------------
;; Scenario: feed a model from a pipeline
;; ---------------------------------------------------------------------------

;; README:begin pipeline
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
;; README:end pipeline

;; ---------------------------------------------------------------------------
;; Error handling
;; ---------------------------------------------------------------------------

;; README:begin errors
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
;; README:end errors

(defn load-missing
  "Loading a model that is not there: the :type of the rejection."
  []
  (-> (models/load! "models/missing")
      (p/then (fn [_] :loaded))
      (p/catch (fn [e] (:type (ex-data e))))))

;; ---------------------------------------------------------------------------
;; Runner
;; ---------------------------------------------------------------------------

(defn- show [title value]
  (println (str ";; " title))
  (pprint/pprint value))

(defn main [& _]
  (-> (p/let [before   (tf/num-tensors)
              tensors  (quick-start)
              trained  (train-xor)
              saved    (save-offline)
              piped    (train-from-pipeline)
              missing  (load-missing)]
        (show "quick start: tensors" tensors)
        (show "train a model" trained)
        (show "save a model, load it back" saved)
        (show "feed a model from a pipeline" piped)
        (show "a product" (attempt #(product [[1 2] [3 4]] [[5 6] [7 8]])))
        (show "shapes that do not multiply" (attempt #(product [[1 2] [3 4]] [[1 2 3]])))
        (show "rows of different lengths" (attempt #(product [[1 2] [3]] [[1] [2]])))
        (show "a model that is not there" missing)
        (show "tensors left behind" (- (tf/num-tensors) before)))
      (p/catch (fn [e]
                 (println "example failed:" (ex-message e) (pr-str (ex-data e)))
                 (set! (.-exitCode js/process) 1)))))
