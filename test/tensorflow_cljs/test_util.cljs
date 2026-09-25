(ns tensorflow-cljs.test-util
  "Shared fixtures for the tensorflow-cljs tests. Not a test namespace itself
   (the name does not end in -test): rejection / thrown capture, approximate
   float32 comparison, an async-test finisher and a temp-dir helper (the
   tests may touch node `os` / `fs` / `path`; the library itself confines
   interop to tensorflow-cljs.core)."
  (:require [cljs.test :refer [is]]
            [promesa.core :as p]))

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

(defn finish
  "Terminates an (async done ..) test: an unexpected rejection of `promise`
   fails one assertion with the reason, then `done` is called."
  [promise done]
  (-> promise
      (p/catch (fn [e] (is false (str "unexpected rejection: " (ex-message e) " " (pr-str (ex-data e))))))
      (p/then (fn [_] (done)))))

(def ^:private os (js/require "os"))
(def ^:private fs (js/require "fs"))
(def ^:private path (js/require "path"))

(defn temp-dir!
  "A fresh temporary directory (os.tmpdir + mkdtemp)."
  [prefix]
  (.mkdtempSync fs (.join path (.tmpdir os) prefix)))

(defn rm-rf!
  "Recursively deletes a directory (best effort)."
  [dir]
  (.rmSync fs dir #js {:recursive true :force true}))

(defn write-file!
  "Writes a string to a file."
  [file content]
  (.writeFileSync fs file content))

(defn write-float32-file!
  "Writes a vector of numbers as little-endian float32 bytes (a fabricated
   weights.bin)."
  [file floats]
  (.writeFileSync fs file (js/Buffer.from (.-buffer (js/Float32Array. (to-array floats))))))

(defn read-file
  "Reads a file as a string."
  [file]
  (.readFileSync fs file "utf8"))

(defn file-exists?
  "True when the path exists."
  [file]
  (.existsSync fs file))

(defn join-path
  "path.join of the segments."
  [& segs]
  (apply (.-join path) segs))

(defn split-binary-file!
  "Splits a binary file into two files at `byte-offset` (for the 2-shard
   manifest test)."
  [file a b byte-offset]
  (let [buf (.readFileSync fs file)]
    (.writeFileSync fs a (.subarray buf 0 byte-offset))
    (.writeFileSync fs b (.subarray buf byte-offset))))
