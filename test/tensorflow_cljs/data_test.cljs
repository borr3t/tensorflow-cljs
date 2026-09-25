(ns tensorflow-cljs.data-test
  "tf.data pipelines: sources, combinators, consumers (node, no network)."
  (:require [cljs.test :refer [deftest is testing async]]
            [promesa.core :as p]
            [tensorflow-cljs.core :as tf]
            [tensorflow-cljs.data :as data]
            [tensorflow-cljs.test-util :as tu]
            [tensorflow-cljs.util :as util]))

(deftest from-vec-batch-take-pipeline
  (async done
    (tu/finish
     (let [ds (data/from-vec [1 2 3 4 5])]
       (is (data/dataset? ds))
       (is (not (data/dataset? [1 2])))
       (is (= 5 (data/size ds)))
       (is (= 3 (data/size (data/batch ds 2))) "5 elements -> 3 batches of up to 2")
       (is (= 2 (data/size (data/batch ds 2 false))) "small last batch dropped")
       (p/let [all (data/to-vec! ds)
               two (data/to-vec! (data/take ds 2))
               batches (data/to-vec! (data/batch ds 2))
               rows (data/to-vec! (data/batch (data/from-vec [{:xs [1 2] :ys 0} {:xs [3 4] :ys 1}]) 2))]
         (is (= [1 2 3 4 5] all))
         (is (= [1 2] two))
         (is (= 3 (count batches)))
         (is (tf/tensor? (first batches)) "batching numbers yields tensors")
         (is (= [1 2] (tf/->vec (first batches))))
         (is (= [5] (tf/->vec (last batches))))
         (is (= 1 (count rows)))
         (is (= #{:xs :ys} (set (keys (first rows)))) "element keys become keywords")
         (is (= [2 2] (tf/shape (:xs (first rows)))) "column-wise stacking")
         (is (= [0 1] (tf/->vec (:ys (first rows)))))
         (data/dispose-all! rows)
         (is (tf/disposed? (:xs (first rows))))))
     done)))

(deftest map-and-filter
  (async done
    (tu/finish
     (let [ds (data/from-vec [1 2 3 4])]
       (p/let [inc'd (data/to-vec! (data/map ds inc))
               odds (data/to-vec! (data/filter ds odd?))
               mapped (data/to-vec! (data/map (data/from-vec [{:xs [1 2] :ys 0}]) (fn [{:keys [xs ys]}] {:xs (tf/tensor xs) :ys (* 2 ys) :tag :a})))
               async'd (data/to-vec! (data/map! ds (fn [x] (p/resolved (* 10 x)))))]
         (is (= [2 3 4 5] inc'd))
         (is (= [1 3] odds))
         (is (nil? (data/size (data/filter ds odd?))) "size unknown after filter")
         (is (= 4 (data/size (data/map ds inc))))
         (is (tf/tensor? (:xs (first mapped))) "map fns may return tensors inside maps")
         (is (= 0 (:ys (first mapped))))
         (is (= "a" (:tag (first mapped))) "keywords become strings")
         (is (= [10 20 30 40] async'd))))
     done)))

(deftest shuffle-repeat-skip-concatenate
  (async done
    (tu/finish
     (let [ds (data/from-vec [1 2 3 4 5 6])]
       (is (infinite? (data/size (data/repeat ds))) "unbounded repeat")
       (is (= 12 (data/size (data/repeat ds 2))))
       (p/let [s1 (data/to-vec! (data/shuffle ds 6 {:seed "x" :reshuffle-each-iteration false}))
               s2 (data/to-vec! (data/shuffle ds 6 {:seed "x" :reshuffle-each-iteration false}))
               rep (data/to-vec! (data/repeat ds 2))
               skipped (data/to-vec! (data/skip ds 4))
               cat (data/to-vec! (data/concatenate (data/from-vec [1]) (data/from-vec [2 3])))
               pre (data/to-vec! (data/prefetch ds 2))
               limited (data/to-vec! (data/take (data/repeat ds) 8))]
         (is (= (set s1) #{1 2 3 4 5 6}))
         (is (= s1 s2) "same seed, same order")
         (is (= 12 (count rep)))
         (is (= [5 6] skipped))
         (is (= [1 2 3] cat))
         (is (= [1 2 3 4 5 6] pre))
         (is (= [1 2 3 4 5 6 1 2] limited))
         (is (= 10000 (data/max-buffer-size)))))
     done)))

(deftest zip-map-and-vector-forms
  (async done
    (tu/finish
     (p/let [m (data/to-vec! (data/zip {:a (data/from-vec [1 2]) :b (data/from-vec [3 4])}))
             v (data/to-vec! (data/zip [(data/from-vec [1 2]) (data/from-vec [3 4])]))]
       (is (= [{:a 1 :b 3} {:a 2 :b 4}] m))
       (is (= [[1 3] [2 4]] v)))
     done)))

(deftest generator-and-func-sources
  (async done
    (tu/finish
     (let [gen (data/generator (fn [] (map (fn [i] {:xs [i] :ys i}) (range 3))))
           counter (atom 0)
           fds (data/func (fn [] (let [i (swap! counter inc)] (when (<= i 3) (* i i)))))]
       (is (nil? (data/size gen)))
       (p/let [g1 (data/to-vec! gen)
               g2 (data/to-vec! (data/batch gen 3))
               f1 (data/to-vec! fds)
               lazy-taken (data/to-vec! (data/take (data/generator (fn [] (range))) 4))]
         (is (= [{:xs [0] :ys 0} {:xs [1] :ys 1} {:xs [2] :ys 2}] g1))
         (is (= [3 1] (tf/shape (:xs (first g2)))) "generator re-runs on every iteration")
         (is (= [1 4 9] f1))
         (is (= [0 1 2 3] lazy-taken) "infinite lazy seq consumed lazily")))
     done)))

(deftest csv-file-source
  (async done
    (let [dir (tu/temp-dir! "tfcljs-csv-")
          file (tu/join-path dir "data.csv")]
      (tu/write-file! file "a,b,label\n1,2,0\n3,4,1\n")
      (tu/finish
       (let [ds (data/csv file {:column-configs {"label" {:is-label true}}})
             flat (data/csv (str "file://" file))]
         (p/let [cols (data/column-names! ds)
                 rows (data/to-vec! ds)
                 flat-rows (data/to-vec! flat)
                 batches (data/to-vec! (data/batch (data/map ds (fn [{:keys [xs ys]}] {:xs [(:a xs) (:b xs)] :ys (:label ys)})) 2))]
           (is (= ["a" "b" "label"] cols))
           (is (= [{:xs {:a 1 :b 2} :ys {:label 0}} {:xs {:a 3 :b 4} :ys {:label 1}}] rows))
           (is (= [{:a 1 :b 2 :label 0} {:a 3 :b 4 :label 1}] flat-rows) "no label config -> flat rows")
           (is (= [[1 2] [3 4]] (tf/->vec (:xs (first batches)))))
           (is (= [0 1] (tf/->vec (:ys (first batches)))))))
       (fn [] (tu/rm-rf! dir) (done))))))

(deftest for-each-and-iterator
  (async done
    (tu/finish
     (let [ds (data/from-vec [1 2 3])
           seen (atom [])]
       (p/let [r (data/for-each! ds (fn [x] (swap! seen conj x) (p/resolved nil)))
               it (data/iterator! ds)
               n1 (data/next! it)
               n2 (data/next! it)
               n3 (data/next! it)
               n4 (data/next! it)]
         (is (nil? r))
         (is (= [1 2 3] @seen))
         (is (= {:value 1 :done false} n1))
         (is (= {:value 2 :done false} n2))
         (is (= {:value 3 :done false} n3))
         (is (true? (:done n4)))
         (is (nil? (:value n4)))))
     done)))

(deftest ->batches-materialises-training-batches
  (async done
    (tu/finish
     (let [before (tf/num-tensors)
           ds (data/from-vec (mapv (fn [i] {:xs [i i] :ys [i]}) (range 5)))]
       (p/let [batches (data/->batches! ds 2)]
         (is (= 3 (count batches)))
         (is (= [2 2] (tf/shape (:xs (first batches)))))
         (is (= [1 1] (tf/shape (:ys (last batches)))))
         (is (= [[4 4]] (tf/->vec (:xs (last batches)))))
         (is (= (+ before 6) (tf/num-tensors)) "3 batches x 2 tensors owned by the caller")
         (data/dispose-all! batches)
         (is (= before (tf/num-tensors)))))
     done)))

(deftest string-elements-and-errors
  (async done
    (tu/finish
     (p/let [strs (data/to-vec! (data/batch (data/from-vec ["a" "b"]) 2))
             e (tu/rejection (data/to-vec! (data/map (data/from-vec [1]) (fn [_] (throw (ex-info "boom" {:user true}))))))]
       (is (= :string (tf/dtype (first strs))))
       (is (= ["a" "b"] (tf/->vec (first strs))))
       (is (= {:user true} (ex-data e)) "consumer ex-infos travel through unchanged"))
     done)))

(deftest size-semantics-and-tensor-elements
  (async done
    (tu/finish
     (let [ds (data/from-vec [(tf/tensor [1 2]) (tf/tensor [3 4])])]
       (is (= 2 (data/size ds)))
       (is (= 1 (data/size (data/batch ds 2))))
       (is (= 1 (data/size (data/take ds 1))))
       (is (= 1 (data/size (data/skip ds 1))))
       (is (= 4 (data/size (data/concatenate ds ds))))
       (is (nil? (data/size (data/func (constantly nil)))))
       (p/let [els (data/to-vec! ds)
               batched (data/to-vec! (data/batch ds 2))
               mapped (data/to-vec! (data/map ds (fn [t] (tf/->vec (tf/mul t 10)))))]
         (is (every? tf/tensor? els) "tensor elements stay opaque")
         (is (= [[1 2] [3 4]] (mapv tf/->vec els)))
         (is (= [[1 2] [3 4]] (tf/->vec (first batched))) "tensor elements stack into one batch tensor")
         (is (= [[10 20] [30 40]] mapped) "map fns can read tensors out")))
     done)))
