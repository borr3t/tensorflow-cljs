(ns tensorflow-cljs.smoke
  "Phase 4 LIVE smoke: the whole library doing REAL compute on the pure-JS
   CPU backend and REAL file I/O under a fresh temporary directory
   (os.tmpdir + mkdtemp, removed at the end). No network anywhere; not part
   of run-tests.sh.

     npx shadow-cljs compile smoke && node target/smoke.js

   Structure (house pattern, see sqlite-cljs / leveldb-cljs): numbered
   sections that never abort early. Each prints `PASS <name> — <detail>` or
   `FAIL <name>: <reason>`, the tally comes last, and the exit code is 0
   only when every section passed. A section that throws or rejects fails
   with the tensorflow-cljs error type / code and the suite continues.

   Sections: 0 backend + platform report, 1 tensor creation + data->clj
   round trip (dtype / ragged rejection), 2 op algebra spot checks + memory
   after tidy, 3 XOR MLP to 100 % + linear regression convergence, 4 save!
   / load! round trip (predictions identical, :include-optimizer reload,
   fabricated 2-shard manifest, fabricated graph model), 5 callbacks
   invoked during fit, 6 early stopping stops early + stop-training!,
   7 data pipeline (csv file -> map -> shuffle -> batch -> channel ->
   fit-dataset!), 8 dispose-all! leaves the memory report clean."
  (:require [clojure.string :as str]
            [promesa.core :as p]
            [tensorflow-cljs.core :as tf]
            [tensorflow-cljs.data :as data]
            [tensorflow-cljs.layers :as layers]
            [tensorflow-cljs.models :as models]
            [tensorflow-cljs.util :as util]))

;; ---------------------------------------------------------------------------
;; Node modules (dev-only interop: no ^js hints, js-invoke like core)
;; ---------------------------------------------------------------------------

(def ^:private fs (js/require "node:fs"))
(def ^:private os (js/require "node:os"))
(def ^:private node-path (js/require "node:path"))

(defn- join-path [& parts] (apply js-invoke node-path "join" parts))
(defn- path-exists? [p] (boolean (js-invoke fs "existsSync" p)))
(defn- mkdtemp! [prefix] (js-invoke fs "mkdtempSync" (join-path (js-invoke os "tmpdir") prefix)))
(defn- rm-rf! [p] (js-invoke fs "rmSync" p #js {:recursive true :force true}))
(defn- write-file! [file content] (js-invoke fs "writeFileSync" file content))
(defn- mkdir! [dir] (js-invoke fs "mkdirSync" dir #js {:recursive true}))
(defn- read-file [file] (js-invoke fs "readFileSync" file "utf8"))
(defn- file-size [file] (unchecked-get (js-invoke fs "statSync" file) "size"))
(defn- write-float32-file! [file floats]
  (js-invoke fs "writeFileSync" file (js/Buffer.from (unchecked-get (js/Float32Array. (to-array floats)) "buffer"))))
(defn- split-binary-file!
  "Splits `file` into `a` (first `at` bytes) and `b` (the rest)."
  [file a b at]
  (let [buf (js-invoke fs "readFileSync" file)]
    (js-invoke fs "writeFileSync" a (js-invoke buf "subarray" 0 at))
    (js-invoke fs "writeFileSync" b (js-invoke buf "subarray" at))))
(defn- now-ms [] (js/Date.now))

;; ---------------------------------------------------------------------------
;; Bookkeeping
;; ---------------------------------------------------------------------------

(def results
  "[{:name .. :ok? .. :reason ..} ...] in section order."
  (atom []))

(def shared
  "Cross-section values: :dir, :start-tensors, :models (to dispose), :tensors,
   :known-leaks [{:count n :reason ..}] (tensors tfjs itself leaks, verified
   by probe, tolerated by section 8)."
  (atom {:models [] :tensors [] :known-leaks []}))

(defn- track-model! [m] (swap! shared update :models conj m) m)
(defn- track-tensors! [& ts] (swap! shared update :tensors into ts) (last ts))
(defn- known-leak! [n reason] (when (pos? n) (swap! shared update :known-leaks conj {:count n :reason reason})) n)

(defn say [& xs] (println (str/join " " (map str xs))))

(defn error-brief
  "One line describing a failure: the tensorflow-cljs error type, :code and
   message. With TFCLJS_SMOKE_STACK=1 in the environment the JS stack of
   the cause (or of the error itself) is appended."
  [e]
  (let [d (or (util/error-data e) (ex-data e))
        stack (when (= "1" (unchecked-get js/process.env "TFCLJS_SMOKE_STACK"))
                (some-> (or (:cause d) e) (unchecked-get "stack") (str/split #"\n") (->> (take 6) (str/join "\n      "))))]
    (str (if d
           (str (name (or (:type d) :error))
                (when (:code d) (str " code=" (name (:code d))))
                (when (:name d) (str " js=" (:name d)))
                " — " (or (ex-message e) (:message d)))
           (or (ex-message e) (some-> e (unchecked-get "message")) (str e)))
         (when stack (str "\n      " stack)))))

(defn section!
  "Runs `f` (a fn returning a promise of a detail string, or throwing /
   rejecting) as section `name`. Prints PASS/FAIL with the elapsed time,
   records the result and ALWAYS resolves so the next section runs."
  [name f]
  (let [t0 (now-ms)
        n0 (tf/num-tensors)
        stamp (fn [] (str " [" (- (now-ms) t0) " ms, tensors " n0 " -> " (tf/num-tensors) "]"))]
    (-> (p/do (f))
        (p/then (fn [detail]
                  (say (str "PASS " name (stamp) (when detail (str " — " detail))))
                  (swap! results conj {:name name :ok? true})))
        (p/catch (fn [e]
                   (let [reason (error-brief e)]
                     (say (str "FAIL " name (stamp) ": " reason))
                     (swap! results conj {:name name :ok? false :reason reason})))))))

(defn fail!
  "Throws a plain smoke failure with `reason`."
  [reason & [data]]
  (throw (ex-info reason (merge {:type :smoke/fail} data))))

(defn check!
  "Fails with `what` unless `ok?`."
  [ok? what]
  (when-not ok? (fail! what)))

(defn thrown
  "The value thrown by (f), nil when it returns normally."
  [f]
  (try (f) nil (catch :default e e)))

(defn rejection
  "Resolves to the error of a rejected promise, nil for a resolved one (the
   error travels inside an already-resolved promise: promesa would turn a
   bare error returned from a handler into a rejection)."
  [promise]
  (-> promise (p/then (fn [_] nil)) (p/catch (fn [e] (p/resolved e)))))

(defn approx=
  "Numbers (or nested sequences of numbers) equal within `eps`."
  ([a b] (approx= a b 1e-5))
  ([a b eps]
   (cond
     (and (number? a) (number? b)) (< (js/Math.abs (- a b)) eps)
     (and (sequential? a) (sequential? b)) (and (= (count a) (count b)) (every? true? (map #(approx= %1 %2 eps) a b)))
     :else (= a b))))

(defn- fmt [x] (if (number? x) (.toFixed x 4) (pr-str x)))

(defn- predict-vec
  "->vec of a prediction, disposing the output tensor."
  [m x]
  (let [o (models/predict m x)] (try (tf/->vec o) (finally (tf/dispose! o)))))

(defn- graph-vec
  "->vec of a graph prediction / execution result, disposing it."
  [f]
  (let [o (f)] (try (tf/->vec o) (finally (tf/dispose! o)))))

;; ---------------------------------------------------------------------------
;; Fixtures
;; ---------------------------------------------------------------------------

(def xor-x-data [[0 0] [0 1] [1 0] [1 1]])
(def xor-y-data [[0] [1] [1] [0]])

(defn- xor-model
  "The seeded XOR MLP of the test suite (glorot seed 7, adam 0.1)."
  []
  (-> (models/sequential [(layers/dense {:units 8 :input-shape [2] :activation :tanh
                                         :kernel-initializer {:type :glorot-uniform :seed 7}})
                          (layers/dense {:units 1 :activation :sigmoid
                                         :kernel-initializer {:type :glorot-uniform :seed 11}})])
      (models/compile-model! {:optimizer {:type :adam :learning-rate 0.1}
                              :loss :binary-crossentropy :metrics [:accuracy]})
      track-model!))

(defn- write-graph-model!
  "Fabricates a tensorflowjs_converter-style graph model y = x * [2 3] + 1
   (weights read from weights.bin) in `dir`."
  [dir]
  (mkdir! dir)
  (write-file!
   (join-path dir "model.json")
   (str "{\"format\":\"graph-model\",\"generatedBy\":\"smoke\",\"convertedBy\":null,"
        "\"modelTopology\":{\"node\":["
        "{\"name\":\"x\",\"op\":\"Placeholder\",\"attr\":{\"dtype\":{\"type\":\"DT_FLOAT\"},\"shape\":{\"shape\":{\"dim\":[{\"size\":\"-1\"},{\"size\":\"2\"}]}}}},"
        "{\"name\":\"w\",\"op\":\"Const\",\"attr\":{\"dtype\":{\"type\":\"DT_FLOAT\"},\"value\":{\"tensor\":{\"dtype\":\"DT_FLOAT\",\"tensorShape\":{\"dim\":[{\"size\":\"2\"}]}}}}},"
        "{\"name\":\"b\",\"op\":\"Const\",\"attr\":{\"dtype\":{\"type\":\"DT_FLOAT\"},\"value\":{\"tensor\":{\"dtype\":\"DT_FLOAT\",\"tensorShape\":{\"dim\":[]}}}}},"
        "{\"name\":\"mul\",\"op\":\"Mul\",\"input\":[\"x\",\"w\"],\"attr\":{\"T\":{\"type\":\"DT_FLOAT\"}}},"
        "{\"name\":\"y\",\"op\":\"Add\",\"input\":[\"mul\",\"b\"],\"attr\":{\"T\":{\"type\":\"DT_FLOAT\"}}}],"
        "\"versions\":{\"producer\":1}},"
        "\"signature\":{\"inputs\":{\"x\":{\"name\":\"x:0\",\"dtype\":\"DT_FLOAT\",\"tensorShape\":{\"dim\":[{\"size\":\"-1\"},{\"size\":\"2\"}]}}},"
        "\"outputs\":{\"y\":{\"name\":\"y:0\",\"dtype\":\"DT_FLOAT\"}}},"
        "\"weightsManifest\":[{\"paths\":[\"./weights.bin\"],\"weights\":[{\"name\":\"w\",\"shape\":[2],\"dtype\":\"float32\"},{\"name\":\"b\",\"shape\":[],\"dtype\":\"float32\"}]}]}"))
  (write-float32-file! (join-path dir "weights.bin") [2 3 1]))

;; ---------------------------------------------------------------------------
;; Sections
;; ---------------------------------------------------------------------------

(defn s0-backend! []
  (p/let [_ (tf/ready!)]
    (let [backend (tf/backend)
          v (tf/version)
          mem (tf/memory)
          dir (mkdtemp! "tfcljs-smoke-")]
      (swap! shared assoc :dir dir :start-tensors (:num-tensors mem))
      (check! (= "cpu" backend) (str "backend is " (pr-str backend) ", expected \"cpu\""))
      (check! (tf/find-backend "cpu") "cpu backend not registered")
      (check! (not (tf/find-backend "webgl")) "webgl unexpectedly registered under node")
      (check! (= "4.22.0" (:tfjs v)) (str "tfjs version " (:tfjs v)))
      (check! (true? (tf/env-flag :is-node)) "IS_NODE flag not true")
      (check! (false? (tf/env-flag :prod)) "PROD flag set — the banner / shape checks would be off")
      (check! (path-exists? dir) "temp dir missing")
      (str "backend=" backend " tfjs=" (:tfjs v) " core=" (:tfjs-core v) " node=" js/process.version
           " platform=" js/process.platform "/" js/process.arch
           " memory start: tensors=" (:num-tensors mem) " bytes=" (:num-bytes mem) " unreliable=" (:unreliable mem)
           " tmp=" dir))))

(defn s1-tensors! []
  (let [a (tf/tensor [[1 2] [3 4]])
        i (tf/tensor [1.9 -2.1 2147483648] :int32)
        b (tf/tensor [true false true])
        s (tf/tensor ["a" "bc"])
        f (tf/tensor [0.1])
        r (tf/tensor [1 2 3 4 5 6] {:shape [2 3]})
        sc (tf/scalar 7)
        ragged (thrown #(tf/tensor [1 [2 3]]))
        ragged2 (thrown #(tf/tensor [[1 2] [3 4 5] [6]]))
        bad-dtype (thrown #(tf/tensor [1 2] :float64))
        bad-shape (thrown #(tf/tensor [1 2 3] {:shape [2 2]}))
        nil-values (thrown #(tf/tensor nil))]
    (track-tensors! a i b s f r sc)
    (check! (= [2 2] (tf/shape a)) (str "shape " (tf/shape a)))
    (check! (= :float32 (tf/dtype a)) "numbers must infer float32")
    (check! (= [[1 2] [3 4]] (tf/->vec a)) "->vec round trip")
    (check! (= [1 2 3 4] (tf/data-sync a)) "data-sync flat")
    (check! (= 4 (tf/get-element a 1 1)) "get-element")
    (check! (= [2] (tf/strides a)) (str "strides " (tf/strides a) " (tfjs omits the trailing 1)"))
    (check! (= :int32 (tf/dtype i)) "explicit :int32")
    (check! (= [1 -2 -2147483648] (tf/->vec i)) (str ":int32 truncates and wraps: " (tf/->vec i)))
    (check! (= :bool (tf/dtype b)) "booleans infer bool")
    (check! (= [true false true] (tf/->vec b)) (str "bool reads back as booleans, not 1/0: " (tf/->vec b)))
    (check! (= :string (tf/dtype s)) "strings infer string")
    (check! (= ["a" "bc"] (tf/->vec s)) "string round trip")
    (check! (not= 0.1 (first (tf/->vec f))) "0.1 must NOT survive float32 exactly")
    (check! (approx= 0.1 (first (tf/->vec f)) 1e-7) "0.1 within float32 precision")
    (check! (= [[1 2 3] [4 5 6]] (tf/->vec r)) ":shape reshapes flat data")
    (check! (= 7 (tf/->vec sc)) "scalar ->vec is the bare value")
    (check! (= :tensorflow-cljs/argument-error (util/error-type ragged)) (str "ragged [1 [2 3]] must be :argument-error, got " (error-brief ragged)))
    (check! (= :tensorflow-cljs/argument-error (util/error-type ragged2)) "ragged [[1 2] [3 4 5] [6]] must be :argument-error")
    (check! (= :tensorflow-cljs/argument-error (util/error-type bad-dtype)) ":float64 must be :argument-error")
    (check! (= :tensorflow-cljs/argument-error (util/error-type bad-shape)) "3 values into shape [2 2] must be :argument-error")
    (check! (= :tensorflow-cljs/argument-error (util/error-type nil-values)) "nil values must be :argument-error")
    (p/let [d (tf/data! a)
            v (tf/->vec! b)]
      (check! (= [1 2 3 4] d) "data! promise")
      (check! (= [true false true] v) "->vec! promise")
      (str "float32 default, 0.1 -> " (first (tf/->vec f)) "; :int32 [1.9 -2.1 2^31] -> " (tf/->vec i)
           "; bool -> " (tf/->vec b) "; string -> " (tf/->vec s)
           "; ragged x2 / :float64 / shape mismatch / nil -> argument-error"))))

(defn s2-ops! []
  (let [a (tf/tensor [[1 2] [3 4]])
        b (tf/tensor [[5 6] [7 8]])
        mm (tf/mat-mul a b)
        r6 (tf/range 0 6)
        rs (tf/reshape r6 [2 3])
        sum-all (tf/sum a)
        sum-ax (tf/sum a 1)
        bc (tf/add a 1)
        eq (tf/equal a 2)
        t3 (tf/tensor [1 5 3])
        tk (tf/topk t3 2)
        cast (tf/cast eq :float32)
        i7 (tf/tensor [7] :int32)
        i2 (tf/tensor [2] :int32)
        int-div (tf/div i7 i2)
        bad (tf/tensor [[1 2 3]])
        mismatch (thrown #(tf/mat-mul a bad))
        disposed (tf/tensor [1])]
    (track-tensors! a b mm r6 rs sum-all sum-ax bc eq cast int-div t3 i7 i2 bad (:values tk) (:indices tk))
    (check! (= [[19 22] [43 50]] (tf/->vec mm)) (str "matmul " (tf/->vec mm)))
    (check! (= [[0 1 2] [3 4 5]] (tf/->vec rs)) (str "reshape " (tf/->vec rs)))
    (check! (= 10 (tf/->vec sum-all)) "sum all")
    (check! (= [3 7] (tf/->vec sum-ax)) "sum axis 1")
    (check! (= [[2 3] [4 5]] (tf/->vec bc)) "broadcast add")
    (check! (= [[false true] [false false]] (tf/->vec eq)) "equal -> bool")
    (check! (= [5 3] (tf/->vec (:values tk))) "topk values")
    (check! (= [1 2] (tf/->vec (:indices tk))) "topk indices")
    (check! (= [[0 1] [0 0]] (tf/->vec cast)) "bool cast to float32 is 1.0 / 0.0")
    (check! (= [3] (tf/->vec int-div)) "int32 / int32 truncates")
    (check! (= :int32 (tf/dtype int-div)) "int32 division stays int32")
    (check! (= :shape-error (util/error-code mismatch)) (str "matmul inner mismatch must be :shape-error, got " (error-brief mismatch)))
    ;; memory: tidy disposes everything but the return value
    (tf/dispose! disposed)
    (let [use-disposed (thrown #(tf/->vec disposed))
          before (tf/num-tensors)
          kept (tf/tidy (fn [] (let [x (tf/tensor [[1 2] [3 4]])
                                    y (tf/tensor [[1 0] [0 1]])
                                    ts (doall (map (fn [_] (tf/mat-mul x y)) (range 10)))]
                                {:result (tf/add (last ts) 0) :count (count ts)})))
          after-tidy (tf/num-tensors)]
      (check! (= 10 (:count kept)) "tidy body ran")
      (check! (= (inc before) after-tidy) (str "tidy must leave exactly the returned tensor: " before " -> " after-tidy))
      (check! (= [[1 2] [3 4]] (tf/->vec (:result kept))) "kept tensor readable")
      (tf/dispose! kept)
      (check! (= before (tf/num-tensors)) "dispose! of the tidy result restores the count")
      (check! (= :tensorflow-cljs/tf-error (util/error-type use-disposed)) "using a disposed tensor is a tf-error")
      (check! (tf/disposed? disposed) "disposed? true")
      (str "matmul " (tf/->vec mm) ", reshape " (tf/->vec rs) ", sum " (tf/->vec sum-all) "/" (tf/->vec sum-ax)
           ", broadcast/equal/topk/cast/int-div ok; tidy: " before " -> " after-tidy " -> " (tf/num-tensors)
           " tensors (12 created inside, 1 kept, then disposed); matmul mismatch -> " (name (util/error-code mismatch))))))

(defn s3-training! []
  (let [xs (track-tensors! (tf/tensor xor-x-data))
        ys (track-tensors! (tf/tensor xor-y-data))
        m (xor-model)
        ;; linear regression y = 3x + 1 on 32 points
        lin-x (track-tensors! (tf/tensor (mapv (fn [i] [(/ i 16)]) (range 32))))
        lin-y (track-tensors! (tf/tensor (mapv (fn [i] [(+ 1 (* 3 (/ i 16)))]) (range 32))))
        lm (-> (models/sequential [(layers/dense {:units 1 :input-shape [1]
                                                  :kernel-initializer {:type :zeros}})])
               (models/compile-model! {:optimizer {:type :sgd :learning-rate 0.1} :loss :mean-squared-error})
               track-model!)
        t0 (now-ms)]
    (p/let [h (models/fit! m xs ys {:epochs 150 :batch-size 4 :yield-every :never})
            xor-ms (- (now-ms) t0)
            ev (models/evaluate! m xs ys)
            pred (models/predict! m xs)
            lh (models/fit! lm lin-x lin-y {:epochs 60 :batch-size 8 :yield-every :never :shuffle false})]
      (track-tensors! pred)
      (let [losses (get-in h [:history :loss])
            rounded (mapv #(if (> % 0.5) 1 0) (map first (tf/->vec pred)))
            lin-losses (get-in lh [:history :loss])
            increases (count (filter (fn [[a b]] (> b a)) (partition 2 1 lin-losses)))
            [w b] (map (comp first tf/->vec) (models/weights lm))
            [w b] [(if (sequential? w) (first w) w) b]]
        (check! (= 150 (count losses)) "150 epochs recorded")
        (check! (= 1 (:acc ev)) (str "XOR accuracy must be 1, got " (:acc ev) " (final loss " (fmt (last losses)) ")"))
        (check! (= [0 1 1 0] rounded) (str "XOR predictions " rounded))
        (check! (< (last losses) (first losses)) "XOR loss decreased")
        (check! (= 60 (count lin-losses)) "60 regression epochs")
        (check! (< (last lin-losses) (* 0.01 (first lin-losses))) (str "regression loss must fall 100x: " (fmt (first lin-losses)) " -> " (fmt (last lin-losses))))
        (check! (<= increases 6) (str "regression loss must be monotonic-ish: " increases " increases in 59 steps"))
        (check! (approx= 3 w 0.3) (str "slope ~3, got " w))
        (check! (approx= 1 b 0.3) (str "intercept ~1, got " b))
        (str "XOR: 150 epochs in " xor-ms " ms, acc " (:acc ev) ", loss " (fmt (first losses)) " -> " (fmt (last losses))
             ", predictions " rounded "; regression: loss " (fmt (first lin-losses)) " -> " (fmt (last lin-losses))
             " (" increases " increases / 59 steps), w=" (fmt w) " b=" (fmt b))))))

(defn s4-save-load! []
  (let [{:keys [dir]} @shared
        m (first (:models @shared))
        xs (tf/tensor xor-x-data)
        ys (tf/tensor xor-y-data)
        expected (predict-vec m xs)
        plain-dir (join-path dir "xor")
        opt-dir (join-path dir "xor-opt")
        shard-dir (join-path dir "xor-shards")
        graph-dir (join-path dir "graph")]
    (track-tensors! xs ys)
    (p/let [saved (models/save! m plain-dir)
            n-before-opt (tf/num-tensors)
            saved-opt (models/save! m opt-dir {:include-optimizer true})
            opt-leak (known-leak! (- (tf/num-tensors) n-before-opt)
                                  "tfjs 4.22 leaks the optimizer's int32 `iter` scalar on every save with :include-optimizer (probe-verified, 1 per save)")
            _ (models/save! m shard-dir)
            _ (do (split-binary-file! (join-path shard-dir "weights.bin") (join-path shard-dir "a.bin") (join-path shard-dir "b.bin") 40)
                  (write-file! (join-path shard-dir "model.json")
                               (str/replace (read-file (join-path shard-dir "model.json")) "[\"./weights.bin\"]" "[\"a.bin\",\"b.bin\"]"))
                  (write-graph-model! graph-dir))
            m2 (models/load! plain-dir)
            m3 (models/load! (join-path opt-dir "model.json"))
            m4 (models/load! (str "file://" shard-dir))
            g (models/load-graph! graph-dir)
            _ (doseq [x [m2 m3 m4]] (track-model! x))
            p2 (predict-vec m2 xs)
            p3 (predict-vec m3 xs)
            p4 (predict-vec m4 xs)
            missing (rejection (models/load! (join-path dir "nope")))
            wrong (rejection (models/load! graph-dir))
            h3 (models/fit! m3 xs ys {:epochs 1 :yield-every :never})
            need-compile (rejection (models/fit! m2 xs ys {:epochs 1 :yield-every :never}))]
      (let [gx (tf/tensor [[1 1] [2 5]])
            gout (graph-vec #(models/graph-predict g gx))
            gmul (graph-vec #(models/graph-execute g gx ["mul"]))]
        (tf/dispose! gx)
        (check! (path-exists? (join-path plain-dir "model.json")) "model.json written")
        (check! (path-exists? (join-path plain-dir "weights.bin")) "weights.bin written")
        (check! (= "JSON" (get-in saved [:model-artifacts-info :model-topology-type])) "artifacts info")
        (check! (= (* 4 (models/count-params m)) (get-in saved [:model-artifacts-info :weight-data-bytes])) "weight bytes = 4 x params")
        (check! (> (get-in saved-opt [:model-artifacts-info :weight-data-bytes]) (get-in saved [:model-artifacts-info :weight-data-bytes])) ":include-optimizer stores optimizer state")
        (check! (= expected p2) (str "plain reload predictions must be identical: " expected " vs " p2))
        (check! (not (models/compiled? m2)) "plain reload is not compiled")
        (check! (= :tensorflow-cljs/tf-error (util/error-type need-compile)) (str "fit! on an uncompiled reload must reject with a tf-error (RuntimeError), got " (error-brief need-compile)))
        (check! (= :runtime-error (util/error-code need-compile)) (str "code must be :runtime-error, got " (util/error-code need-compile)))
        (check! (models/compiled? m3) ":include-optimizer reload is compiled")
        (check! (= ["loss" "acc"] (models/metrics-names m3)) (str "metrics restored: " (models/metrics-names m3)))
        (check! (= 1 (count (:epoch h3))) "compiled reload trains without compile-model!")
        (check! (= expected p3) (str ":include-optimizer reload predictions must be identical: " p3))
        (check! (= expected p4) (str "2-shard manifest reload predictions must be identical: " p4))
        (check! (= 40 (file-size (join-path shard-dir "a.bin"))) "shard a.bin is 40 bytes")
        (check! (models/graph-model? g) "graph model loaded")
        (check! (= [[3 4] [5 16]] gout) (str "graph predict y = x*[2 3]+1: " gout))
        (check! (= [[2 3] [4 15]] gmul) (str "graph execute mul: " gmul))
        (check! (= ["x"] (models/graph-input-names g)) "graph input names")
        (check! (= :tensorflow-cljs/io-error (util/error-type missing)) (str "missing dir must be io-error: " (error-brief missing)))
        (check! (= :tensorflow-cljs/io-error (util/error-type wrong)) (str "load! of a graph model must be io-error: " (error-brief wrong)))
        (check! (<= opt-leak 1) (str ":include-optimizer save leaked " opt-leak " tensors, tfjs's known leak is 1"))
        (models/graph-dispose! g)
        (str "model.json+weights.bin (" (get-in saved [:model-artifacts-info :weight-data-bytes]) " B), :include-optimizer ("
             (get-in saved-opt [:model-artifacts-info :weight-data-bytes]) " B) reload compiled + trained 1 epoch; plain reload -> "
             (name (util/error-code need-compile)) " without compile; 2-shard manifest (40 B + rest) identical; graph model y=x*[2 3]+1 -> "
             gout "; missing/graph-via-load! -> io-error; :include-optimizer save leaked " opt-leak " tensor (tfjs iter scalar)")))))

(defn s5-callbacks! []
  (let [m (xor-model)
        xs (tf/tensor xor-x-data)
        ys (tf/tensor xor-y-data)
        counts (atom {:train-begin 0 :train-end 0 :epoch-begin 0 :epoch-end 0 :batch-begin 0 :batch-end 0 :yield 0})
        seen-logs (atom [])
        bump (fn [k] (fn [& args] (swap! counts update k inc) (when-let [logs (last args)] (when (map? logs) (swap! seen-logs conj logs))) nil))]
    (track-tensors! xs ys)
    (p/let [h (models/fit! m xs ys {:epochs 3 :batch-size 2 :yield-every 1   ; ms: on-yield only fires for numeric / :auto intervals in tfjs 4.22
                                    :callbacks {:on-train-begin (bump :train-begin) :on-train-end (bump :train-end)
                                                :on-epoch-begin (bump :epoch-begin) :on-epoch-end (bump :epoch-end)
                                                :on-batch-begin (bump :batch-begin) :on-batch-end (bump :batch-end)
                                                :on-yield (bump :yield)}})
            ;; a vector mixing a hook map with an early-stopping object (tfjs would crash on the raw mix)
            h2 (models/fit! m xs ys {:epochs 2 :batch-size 4 :yield-every :never
                                     :callbacks [{:on-epoch-end (fn [_ _] (swap! counts update :mixed (fnil inc 0)) nil)}
                                                 (models/early-stopping {:monitor :loss :patience 5})]})
            ;; a hook returning a promise is awaited
            h3 (models/fit! m xs ys {:epochs 1 :batch-size 4 :yield-every :never
                                     :callbacks {:on-epoch-end (fn [_ _] (p/delay 20 (swap! counts update :async (fnil inc 0))))}})]
      (let [c @counts
            epoch-logs (filter :loss @seen-logs)]
        (check! (= 3 (:epoch-end c)) (str "on-epoch-end must run 3 times, got " (:epoch-end c)))
        (check! (= 3 (:epoch-begin c)) "on-epoch-begin 3")
        (check! (= 6 (:batch-end c)) (str "on-batch-end must run 6 times (3 epochs x 2 batches), got " (:batch-end c)))
        (check! (= 6 (:batch-begin c)) "on-batch-begin 6")
        (check! (= 1 (:train-begin c)) "on-train-begin once")
        (check! (= 1 (:train-end c)) "on-train-end once")
        (check! (pos? (:yield c)) (str "on-yield with :yield-every 1 (ms), got " (:yield c)))
        (check! (seq epoch-logs) "logs carry :loss")
        (check! (every? #(contains? % :acc) epoch-logs) "logs carry :acc (kebab, numbers)")
        (check! (every? number? (map :loss epoch-logs)) "log values are numbers")
        (check! (= 3 (count (:epoch h))) "history epochs")
        (check! (= 2 (:mixed c)) "hook map mixed with early-stopping in one vector")
        (check! (= 2 (count (:epoch h2))) "mixed callbacks fit ran 2 epochs")
        (check! (= 1 (:async c)) "promise-returning hook awaited")
        (check! (= 1 (count (:epoch h3))) "h3 one epoch")
        (str "3 epochs x 2 batches: epoch-end " (:epoch-end c) ", batch-end " (:batch-end c) ", train-begin/end "
             (:train-begin c) "/" (:train-end c) ", yields " (:yield c) " (:yield-every 1 ms, debounced); logs keys " (sort (keys (first epoch-logs)))
             "; hook map + early-stopping mixed ok; async hook awaited")))))

(defn s6-early-stopping! []
  (let [m (xor-model)
        xs (tf/tensor xor-x-data)
        ys (tf/tensor xor-y-data)
        flipped (tf/tensor (mapv (fn [[y]] [(- 1 y)]) xor-y-data))
        m2 (xor-model)]
    (track-tensors! xs ys flipped)
    (p/let [;; obviously overfit: validation labels are the training labels flipped, so val-loss rises as train loss falls
            h (models/fit! m xs ys {:epochs 200 :batch-size 4 :yield-every :never
                                    :validation-data [xs flipped]
                                    :callbacks [(models/early-stopping {:monitor :val-loss :patience 1})]})
            h2 (models/fit! m2 xs ys {:epochs 50 :batch-size 4 :yield-every :never
                                      :callbacks {:on-epoch-end (fn [epoch _] (when (= epoch 2) (models/stop-training! m2)))}})]
      (let [ran (count (:epoch h))
            vl (get-in h [:history :val-loss])
            ran2 (count (:epoch h2))]
        (check! (< ran 200) (str "early stopping must stop before 200 epochs, ran " ran))
        (check! (>= ran 2) (str "at least 2 epochs ran: " ran))
        (check! (= ran (count vl)) "val-loss recorded per epoch")
        (check! (> (last vl) (apply min vl)) "the last val-loss is worse than the best (that is why it stopped)")
        (check! (= 3 ran2) (str "stop-training! at epoch index 2 must end after 3 epochs, ran " ran2))
        (str "early-stopping(:val-loss, patience 1) on flipped validation labels stopped after " ran "/200 epochs (val-loss "
             (fmt (first vl)) " -> " (fmt (last vl)) ", best " (fmt (apply min vl)) "); stop-training! from on-epoch-end: " ran2 "/50 epochs")))))

(defn s7-data! []
  (let [{:keys [dir]} @shared
        file (join-path dir "xor.csv")
        rows (for [i (range 8) :let [[a b] (nth xor-x-data (mod i 4))]] (str a "," b "," (first (nth xor-y-data (mod i 4)))))
        _ (write-file! file (str "x1,x2,label\n" (str/join "\n" rows) "\n"))
        csv (data/csv file {:column-configs {"label" {:is-label true}}})
        ds (-> csv
               (data/map (fn [{:keys [xs ys]}] {:xs [(:x1 xs) (:x2 xs)] :ys [(:label ys)]}))
               (data/shuffle 8 {:seed "smoke"})
               (data/batch 4))
        m (xor-model)]
    (p/let [cols (data/column-names! csv)
            batches (data/->batches! (data/map csv (fn [{:keys [xs ys]}] {:xs [(:x1 xs) (:x2 xs)] :ys [(:label ys)]})) 4)
            from-chan (data/channel->vec! (data/->channel! (data/take ds 2) {:buffer 1}))
            step (models/train-on-batch! m (:xs (first from-chan)) (:ys (first from-chan)))
            h (models/fit-dataset! m (data/repeat ds) {:epochs 2 :batches-per-epoch 2 :yield-every :never})
            size-err (rejection (models/fit-dataset! m ds {:epochs 1}))]
      (check! (= ["x1" "x2" "label"] cols) (str "csv columns " cols))
      (check! (= 2 (count batches)) (str "->batches! of 8 rows by 4 -> 2 batches, got " (count batches)))
      (check! (= [4 2] (tf/shape (:xs (first batches)))) (str "batch xs shape " (tf/shape (:xs (first batches)))))
      (check! (= [4 1] (tf/shape (:ys (first batches)))) (str "batch ys shape " (tf/shape (:ys (first batches)))))
      (check! (= [[0 0] [0 1] [1 0] [1 1]] (tf/->vec (:xs (first batches)))) "csv rows in file order without shuffle")
      (check! (= 2 (count from-chan)) (str "core.async channel delivered 2 batches, got " (count from-chan)))
      (check! (every? #(= [4 2] (tf/shape (:xs %))) from-chan) "channel batches shaped [4 2]")
      (check! (not= (tf/->vec (:xs (first from-chan))) (tf/->vec (:xs (first batches)))) "shuffle changed the order (seeded)")
      (check! (vector? step) (str "train-on-batch! returns [loss acc]: " step))
      (check! (= 2 (count (:epoch h))) "fit-dataset! 2 epochs")
      (check! (nil? (data/size ds)) "csv-derived dataset size is unknown")
      (check! (nil? (data/size (data/repeat ds))) "repeat of an unknown size stays unknown")
      (check! (infinite? (data/size (data/repeat (data/from-vec [1 2])))) "repeat of a known size is ##Inf")
      (check! (= :tensorflow-cljs/argument-error (util/error-type size-err)) "fit-dataset! without :batches-per-epoch on an unknown size -> argument-error")
      (data/dispose-all! batches)
      (data/dispose-all! from-chan)
      (str "csv " (count rows) " rows, columns " cols "; ->batches! 2 x {:xs [4 2] :ys [4 1]}; ->channel! 2 batches (shuffled, seed \"smoke\"); "
           "train-on-batch! " (mapv fmt step) "; fit-dataset! on (repeat ds): 2 epochs x :batches-per-epoch 2, loss "
           (fmt (first (get-in h [:history :loss]))) " -> " (fmt (last (get-in h [:history :loss])))
           "; unknown size without :batches-per-epoch -> argument-error"))))

(defn s8-dispose! []
  (let [{:keys [models tensors start-tensors known-leaks]} @shared
        before (tf/num-tensors)
        disposed (mapv models/dispose! models)
        _ (tf/dispose! tensors)
        _ (tf/dispose-variables!)
        mem (tf/memory)
        tolerated (reduce + 0 (map :count known-leaks))]
    (check! (every? number? (map :num-disposed-variables disposed)) "models/dispose! reports")
    (check! (= (+ start-tensors tolerated) (:num-tensors mem))
            (str "memory report must be back at the start count " start-tensors " plus " tolerated
                 " known tfjs leak(s), got " (:num-tensors mem) " (was " before " before disposal)"))
    (str (count models) " models disposed (" (reduce + (map :num-disposed-variables disposed)) " variables), "
         (count tensors) " tracked tensors disposed; tensors " before " -> " (:num-tensors mem)
         " (= start " start-tensors " + " tolerated " known tfjs leak: " (str/join "; " (map :reason known-leaks)) ")"
         ", bytes " (:num-bytes mem) ", data buffers " (:num-data-buffers mem))))

;; ---------------------------------------------------------------------------
;; Main
;; ---------------------------------------------------------------------------

(defn- cleanup! []
  (when-let [dir (:dir @shared)]
    (rm-rf! dir)
    (say (str "cleanup: removed " dir " (gone=" (not (path-exists? dir)) ")"))))

(defn main [& _]
  (say "tensorflow-cljs LIVE smoke —" (js/Date.) "— pure @tensorflow/tfjs CPU backend, real files in a temp dir, no network")
  (let [t0 (now-ms)]
    (-> (p/do
          (section! "0 backend + platform report" s0-backend!)
          (section! "1 tensor creation + data->clj round trip (dtypes, ragged / dtype rejection)" s1-tensors!)
          (section! "2 op algebra spot checks (matmul / reshape / sum) + memory after tidy" s2-ops!)
          (section! "3 XOR MLP trains to 100 % (seeded) + linear regression converges" s3-training!)
          (section! "4 save! / load! round trip, :include-optimizer, 2-shard manifest, graph model" s4-save-load!)
          (section! "5 callbacks invoked during fit! (hook maps, mixed vector, async hook)" s5-callbacks!)
          (section! "6 early stopping stops early (patience 1, flipped validation) + stop-training!" s6-early-stopping!)
          (section! "7 data pipeline: csv file -> map -> shuffle -> batch -> ->channel! / ->batches! -> train" s7-data!)
          (section! "8 dispose-all! leaves the memory report clean" s8-dispose!))
        (p/catch (fn [e] (say "FAIL harness:" (error-brief e))))
        (p/then (fn [_] (cleanup!)))
        (p/finally (fn [_ _]
                     (let [rs @results
                           failed (remove :ok? rs)]
                       (say "")
                       (say (str "TALLY: " (count (filter :ok? rs)) "/" (count rs) " sections passed in " (- (now-ms) t0) " ms"))
                       (doseq [f failed] (say (str "  FAILED " (:name f) ": " (:reason f))))
                       (js/process.exit (if (and (seq rs) (empty? failed)) 0 1))))))))
