(ns tensorflow-cljs.test-util
  "Shared fixtures for the tensorflow-cljs tests. Not a test namespace itself
   (the name does not end in -test). Phase 2 adds tensor-equality helpers
   (approximate float32 comparison) and a memory-leak fixture (tf.memory()
   numTensors before/after) here so tests stay deterministic and in-process."
  (:require [promesa.core :as p]))

(defn rejection
  "Promise of the rejection reason of `p` (fails the assertion if it resolves)."
  ;; NB: promesa treats a js/Error RETURNED from a catch handler as a
  ;; rejection (IPromiseFactory on js/Error), so the error must be wrapped in
  ;; an already-resolved promise to travel as a value.
  [promise]
  (-> promise
      (p/then (fn [v] (throw (ex-info "expected rejection" {:resolved v}))))
      (p/catch (fn [e] (p/resolved e)))))

(defn thrown
  "The value thrown by (f), nil when it returns normally."
  [f]
  (try (f) nil (catch :default e e)))

(defn approx=
  "True when two numbers (or nested sequences of numbers) agree within
   `eps` (default 1e-5, float32-friendly)."
  ([a b] (approx= a b 1e-5))
  ([a b eps]
   (cond
     (and (number? a) (number? b)) (< (js/Math.abs (- a b)) eps)
     (and (sequential? a) (sequential? b)) (and (= (count a) (count b))
                                                (every? true? (map #(approx= %1 %2 eps) a b)))
     :else (= a b))))
