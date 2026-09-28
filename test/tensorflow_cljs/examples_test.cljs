(ns tensorflow-cljs.examples-test
  "The examples of the README, run. Everything is computed in this process
   on the CPU backend; the save / load example writes in a temporary
   directory and removes it.

   The README is held to the code:
   - what stands between a README:begin and a README:end mark of
     dev/tensorflow_cljs/examples.cljs stands in README.md as written;
   - every one-liner of `shown` stands in README.md with the value it
     evaluates to;
   - every result README.md prints is the result the example returns.
   Spacing and commas are not compared."
  (:require ["fs" :as fs]
            [cljs.test :refer [deftest is testing async]]
            [clojure.string :as str]
            [promesa.core :as p]
            [tensorflow-cljs.core :as tf]
            [tensorflow-cljs.examples :as ex :refer [attempt product]]
            [tensorflow-cljs.models :as models]
            [tensorflow-cljs.test-util :as tu]
            [tensorflow-cljs.util :as util])
  (:require-macros [tensorflow-cljs.doctest :refer [claims]]))

;; ---------------------------------------------------------------------------
;; The one-liners
;; ---------------------------------------------------------------------------

(def shown
  "The one-liners of the README, each with its value."
  (claims
   (tf/tidy (fn [] (tf/->vec (tf/tensor [1.9 -2.1] :int32))))
   (tf/tidy (fn [] (tf/->vec (tf/tensor [0.1]))))
   (tf/tidy (fn [] (tf/->vec (tf/reshape (tf/range 0 6) [2 3]))))
   (tf/tidy (fn [] (tf/->vec (tf/arg-max (tf/tensor [[0.1 0.7 0.2] [0.8 0.1 0.1]]) 1))))
   (tf/tidy (fn [] (tf/shape (tf/one-hot (tf/tensor [0 2] :int32) 3))))
   (tf/tidy (fn [] (tf/->vec ((tf/grad tf/square) (tf/tensor [1 2 3])))))
   (tf/backend)
   (util/->config {:units 4 :kernel-initializer :glorot-uniform})
   (util/enum-value :glorot-uniform)
   (attempt (fn [] (product [[1 2] [3 4]] [[5 6] [7 8]])))
   (attempt (fn [] (product [[1 2] [3 4]] [[1 2 3]])))
   (attempt (fn [] (product [[1 2] [3]] [[1] [2]])))))

;; ---------------------------------------------------------------------------
;; Reading the README
;; ---------------------------------------------------------------------------

(defn- slurp [path]
  (fs/readFileSync path "utf8"))

(defn- plain
  "`text` without what is not compared: commas, and any run of whitespace
   as one space."
  [text]
  (-> text
      (str/replace "," "")
      (str/replace #"\s+" " ")
      str/trim))

(defn- region
  "The text between the marks README:begin `label` and README:end `label`
   of the file at `path`, nil when either mark is missing."
  [path label]
  (let [[_ found] (re-find (re-pattern (str "(?s);; README:begin " label "\\n(.*?)\\n\\s*;; README:end " label))
                           (slurp path))]
    found))

(defn- shows?
  "Does README.md show `text`?"
  [text]
  (str/includes? (plain (slurp "README.md")) (plain text)))

(defn- shown-in-readme
  "Asserts README.md prints `value`."
  [value]
  (is (shows? (pr-str value)) (str "README.md does not show:\n" (pr-str value))))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest the-quick-start-computes
  (let [before (tf/num-tensors)
        r (ex/quick-start)]
    (is (= {:shape [2 2]
            :dtype :float32
            :product [[19 22] [43 50]]
            :plus-one [[2 3] [4 5]]
            :mean 2.5
            :by-row [3 7]
            :is-two? [[false true] [false false]]}
           r))
    (is (= before (tf/num-tensors)) "no tensor is left behind")
    (shown-in-readme r)))

(deftest a-model-is-trained
  (async done
    (tu/finish
     (p/let [before (tf/num-tensors)
             r (ex/train-xor)]
       (is (= {:epochs 150 :loss-fell? true :accuracy 1 :answers [[0] [1] [1] [0]]} r))
       (is (= before (tf/num-tensors)) "the model and its tensors are freed")
       (shown-in-readme r))
     done)))

(deftest an-untrained-model-answers-otherwise
  (let [model (ex/xor-model)]
    (is (not= ex/ys (ex/answers model ex/xs)) "the answers of the README are learned")
    (models/dispose! model)))

(deftest a-model-is-saved-and-loaded
  (async done
    (tu/finish
     (p/let [before (tf/num-tensors)
             r (ex/save-offline)]
       (is (= {:files ["model.json" "weights.bin"] :same-answers? true :compiled? false} r))
       (is (= before (tf/num-tensors)) "both models are freed")
       (shown-in-readme r))
     done)))

(deftest a-model-is-fed-from-a-pipeline
  (async done
    (tu/finish
     (p/let [r (ex/train-from-pipeline)]
       (is (= {:batches [[[2 2] [2 1]] [[2 2] [2 1]]]
               :epochs 150
               :answers [[0] [1] [1] [0]]}
              r))
       (shown-in-readme r))
     done)))

(deftest a-failure-is-data
  (let [before (tf/num-tensors)]
    (is (= {:ok? true :value [[19 22] [43 50]]} (attempt #(product [[1 2] [3 4]] [[5 6] [7 8]]))))
    (is (= {:ok? false :why :shape-error} (attempt #(product [[1 2] [3 4]] [[1 2 3]]))))
    (is (= {:ok? false :why :bad-argument} (attempt #(product [[1 2] [3]] [[1] [2]]))))
    (is (thrown? js/Error (attempt #(throw (js/Error. "not of the library"))))
        "what is not an error of the library is thrown on")
    (is (= before (tf/num-tensors)) "a failure leaves no tensor behind")))

(deftest a-missing-model-rejects
  (async done
    (tu/finish
     (p/let [r (ex/load-missing)]
       (is (= :tensorflow-cljs/io-error r))
       (is (shows? ";; => :tensorflow-cljs/io-error")))
     done)))

(deftest the-one-liners-hold
  (let [value (into {} shown)]
    (is (= (count shown) (count value)) "no one-liner is shown twice")
    (is (= [1 -2] (value "(tf/tidy (fn [] (tf/->vec (tf/tensor [1.9 -2.1] :int32))))")))
    (is (= [[0 1 2] [3 4 5]] (value "(tf/tidy (fn [] (tf/->vec (tf/reshape (tf/range 0 6) [2 3]))))")))
    (is (= [1 0] (value "(tf/tidy (fn [] (tf/->vec (tf/arg-max (tf/tensor [[0.1 0.7 0.2] [0.8 0.1 0.1]]) 1))))")))
    (is (= [2 3] (value "(tf/tidy (fn [] (tf/shape (tf/one-hot (tf/tensor [0 2] :int32) 3))))")))
    (is (= [2 4 6] (value "(tf/tidy (fn [] (tf/->vec ((tf/grad tf/square) (tf/tensor [1 2 3])))))")))
    (is (= "cpu" (value "(tf/backend)")))
    (is (= "glorotUniform" (value "(util/enum-value :glorot-uniform)")))))

(deftest the-readme-shows-what-runs
  (testing "the marked code of the examples"
    (doseq [label ["quick-start" "train" "save" "pipeline" "errors"]
            :let [code (region "dev/tensorflow_cljs/examples.cljs" label)]]
      (is (some? code) (str "no region " label))
      (is (shows? code) (str "README.md does not show " label ":\n" code))))
  (testing "every one-liner, with its value"
    (doseq [[source value] shown
            :let [line (str source "\n;; => " (pr-str value))]]
      (is (shows? line) (str "README.md does not show:\n" line))))
  (testing "how the examples are run, and where the reference is"
    (is (shows? "npx shadow-cljs compile examples && node target/examples.js"))
    (is (shows? "docs/tensorflow-cljs.org"))
    (is (shows? "docs/tensorflow-api-inventory.org"))))
