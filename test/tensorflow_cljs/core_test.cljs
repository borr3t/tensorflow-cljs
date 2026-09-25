(ns tensorflow-cljs.core-test
  "Numeric round trips of the tensor core on the CPU backend (no network)."
  (:require [cljs.test :refer [deftest is testing async]]
            [promesa.core :as p]
            [tensorflow-cljs.core :as tf]
            [tensorflow-cljs.test-util :as tu :refer [approx= thrown]]
            [tensorflow-cljs.util :as util]))

(deftest creation-infers-shape-and-dtype
  (let [t (tf/tensor [[1 2] [3 4]])]
    (is (tf/tensor? t))
    (is (= [2 2] (tf/shape t)))
    (is (= :float32 (tf/dtype t)) "numbers (integers included) -> float32")
    (is (= 2 (tf/rank t)))
    (is (= 4 (tf/size t)))
    (is (= [[1 2] [3 4]] (tf/->vec t)))
    (is (= [[1 2] [3 4]] (tf/->clj t)) "->clj is ->vec"))
  (is (= :bool (tf/dtype (tf/tensor [true false]))))
  (is (= :string (tf/dtype (tf/tensor ["a" "b"]))))
  (is (= [] (tf/shape (tf/tensor 3))))
  (is (= 3 (tf/->vec (tf/scalar 3))) "scalar reads back as a bare value")
  (is (= [3] (tf/shape (tf/tensor1d [1 2 3]))))
  (is (= [1 3] (tf/shape (tf/tensor2d [[1 2 3]]))))
  (is (= [1 1 2] (tf/shape (tf/tensor3d [[[1 2]]]))))
  (is (= [1 1 1 2] (tf/shape (tf/tensor4d [[[[1 2]]]]))))
  (is (not (tf/tensor? [1 2]))))

(deftest explicit-shape-and-dtype
  (let [t (tf/tensor [1 2 3 4] {:shape [2 2] :dtype :int32})]
    (is (= [2 2] (tf/shape t)))
    (is (= :int32 (tf/dtype t)))
    (is (= [[1 2] [3 4]] (tf/->vec t))))
  (is (= :int32 (tf/dtype (tf/tensor [1 2] :int32))) "a bare dtype keyword as opts")
  (is (= [1 -2147483648] (tf/->vec (tf/tensor [1.9 2147483648] :int32))) "int32 truncates and wraps")
  (let [x (tf/->vec (tf/scalar 0.1))]
    (is (not= 0.1 x) "doubles are narrowed to float32")
    (is (approx= 0.1 x 1e-7)))
  (is (= [1 0] (tf/->vec (tf/tensor [true false] :int32))) "booleans as int32")
  (is (= :float32 (tf/dtype (tf/cast (tf/tensor [true false]) :float32))))
  (is (= [1 0] (tf/->vec (tf/cast (tf/tensor [true false]) :float32)))))

(deftest creation-argument-errors
  (let [e (thrown #(tf/tensor [1 2 3] {:shape [2 2]}))]
    (is (= :tensorflow-cljs/argument-error (util/error-type e)) "element count checked before tfjs")
    (is (= 3 (:count (ex-data e)))))
  (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(tf/tensor [1 2] {:shape [2 -1]})))))
  (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(tf/tensor [1] :float64)))))
  (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(tf/tensor nil)))))
  (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(tf/zeros [2 1.5])))))
  (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(tf/tensor2d [1 2] {:shape [2]})))))
  (let [e (thrown #(tf/tensor [1 [2 3]]))]
    (is (= :tensorflow-cljs/argument-error (util/error-type e))
        "ragged input under a scalar first element is caught by the wrapper (tfjs 4.22 accepts it silently)")
    (is (= [2] (:shape (ex-data e)))))
  (let [e (thrown #(tf/tensor [[1 2] [3]]))]
    (is (= :tensorflow-cljs/argument-error (util/error-type e)) "ragged input below an array"))
  (let [e (thrown #(tf/tensor [[1 2] [3 4 5] [6]]))]
    (is (= :tensorflow-cljs/argument-error (util/error-type e))
        "count-compatible ragged input (tfjs 4.22 would silently make it [3 2])")
    (is (= [3 2] (:shape (ex-data e)))))
  (let [e (thrown #(tf/tensor2d [1 2 3]))]
    (is (= :tensorflow-cljs/tf-error (util/error-type e)) "rank check of tensor2d")))

(deftest constant-and-sequence-creation
  (is (= [[0 0 0] [0 0 0]] (tf/->vec (tf/zeros [2 3]))))
  (is (= :int32 (tf/dtype (tf/zeros [1] :int32))))
  (is (= [[1 1]] (tf/->vec (tf/ones [1 2]))))
  (is (= [7 7] (tf/->vec (tf/fill [2] 7))))
  (is (= [[1 0] [0 1]] (tf/->vec (tf/eye 2))))
  (is (= [2 3] (tf/shape (tf/eye 2 {:num-columns 3}))))
  (is (= [[0 0] [0 0]] (tf/->vec (tf/zeros-like (tf/ones [2 2])))))
  (is (= [1 1] (tf/->vec (tf/ones-like (tf/tensor [5 6])))))
  (is (approx= [0 0.5 1] (tf/->vec (tf/linspace 0 1 3))))
  (is (= [0 2 4] (tf/->vec (tf/range 0 5 2))))
  (is (= :int32 (tf/dtype (tf/range 0 3 1 :int32))))
  (is (= [0 1 2] (tf/->vec (tf/range 0 3))))
  (is (= [[1 2] [3 4]] (tf/->vec (tf/clone (tf/tensor [[1 2] [3 4]]))))))

(deftest random-creation-and-one-hot
  (let [a (tf/->vec (tf/random-normal [3] {:seed 42}))
        b (tf/->vec (tf/random-normal [3] {:seed 42}))]
    (is (= 3 (count a)))
    (is (= a b) "seeded draws repeat"))
  (is (= [2 2] (tf/shape (tf/random-normal [2 2] {:mean 5 :std-dev 0}))))
  (is (approx= [5 5] (tf/->vec (tf/random-normal [2] {:mean 5 :std-dev 0}))))
  (is (every? #(and (>= % 0) (< % 1)) (tf/->vec (tf/random-uniform [10]))))
  (let [t (tf/random-uniform-int [5] 0 3)]
    (is (= :int32 (tf/dtype t)))
    (is (every? #{0 1 2} (tf/->vec t))))
  (is (= [4] (tf/shape (tf/truncated-normal [4]))))
  (is (= [2] (tf/shape (tf/random-standard-normal [2] {:seed 1}))))
  (is (= [2] (tf/shape (tf/random-gamma [2] 2))))
  (is (= [9 9] (tf/->vec (tf/rand [2] (constantly 9)))))
  (is (= [[1 0 0] [0 0 1]] (tf/->vec (tf/one-hot [0 2] 3))))
  (is (= :int32 (tf/dtype (tf/one-hot [0 2] 3))))
  (is (= [[0.5 1]] (tf/->vec (tf/one-hot (tf/tensor [1] :int32) 2 {:on-value 1 :off-value 0.5 :dtype :float32}))))
  (is (= :shape-error (util/error-code (thrown #(tf/one-hot (tf/tensor [0.0 1.0]) 2)))) "float indices rejected")
  (is (= [1 2] (tf/shape (tf/multinomial (tf/tensor [[0 100]]) 2)))))

(deftest arithmetic-and-broadcasting
  (let [m (tf/tensor [[1 2] [3 4]])]
    (is (= [[2 3] [4 5]] (tf/->vec (tf/add m 1))) "scalar number broadcasts")
    (is (= [[11 22] [13 24]] (tf/->vec (tf/add m [10 20]))) "row vector broadcasts")
    (is (= [[11 12] [23 24]] (tf/->vec (tf/add m [[10] [20]]))) "column vector broadcasts")
    (is (= [[0 1] [2 3]] (tf/->vec (tf/sub m 1))))
    (is (= [[2 4] [6 8]] (tf/->vec (tf/mul m 2))))
    (is (= [[0.5 1] [1.5 2]] (tf/->vec (tf/div m 2))))
    (is (= [[-1 -2] [-3 -4]] (tf/->vec (tf/neg m))))
    (is (= [[1 4] [9 16]] (tf/->vec (tf/square m))))
    (is (= [[1 8] [27 64]] (tf/->vec (tf/pow m 3)))))
  (is (= 3 (tf/->vec (tf/div (tf/scalar 7 :int32) (tf/scalar 2 :int32)))) "int32 division truncates")
  (is (= :float32 (tf/dtype (tf/add (tf/scalar 1 :int32) (tf/scalar 1.5)))) "int32 + float32 -> float32")
  (is (= [1 2] (tf/->vec (tf/abs (tf/tensor [-1 2])))))
  (is (= [-1 0 1] (tf/->vec (tf/sign (tf/tensor [-3 0 2])))))
  (is (= [2 3] (tf/->vec (tf/sqrt (tf/tensor [4 9])))))
  (is (= [0.5] (tf/->vec (tf/rsqrt (tf/tensor [4])))))
  (is (= [0.25] (tf/->vec (tf/reciprocal (tf/tensor [4])))))
  (is (= [4 9] (tf/->vec (tf/squared-difference (tf/tensor [1 2]) (tf/tensor [3 5])))))
  (is (= [3 4] (tf/->vec (tf/maximum (tf/tensor [1 4]) (tf/tensor [3 2])))))
  (is (= [1 2] (tf/->vec (tf/minimum (tf/tensor [1 4]) (tf/tensor [3 2])))))
  (is (= [0 1 5 5] (tf/->vec (tf/clip-by-value (tf/tensor [-1 1 7 5]) 0 5))))
  (is (= [1 2 -1] (tf/->vec (tf/floor (tf/tensor [1.5 2.9 -0.5])))))
  (is (= [2 3 0] (tf/->vec (tf/ceil (tf/tensor [1.5 2.9 -0.5])))))
  (is (= [2 3] (tf/->vec (tf/round (tf/tensor [1.6 2.6])))))
  (is (= [1 0] (tf/->vec (tf/mod (tf/tensor [7 4]) 2))))
  (is (= [3 -1] (tf/->vec (tf/floor-div (tf/tensor [7 -1]) 2))))
  (is (= [0 2] (tf/->vec (tf/div-no-nan (tf/tensor [1 4]) (tf/tensor [0 2])))))
  (is (= [6 8] (tf/->vec (tf/add-n [(tf/tensor [1 2]) (tf/tensor [2 3]) (tf/tensor [3 3])]))))
  (is (approx= [0.7853982] (tf/->vec (tf/atan2 (tf/tensor [1]) (tf/tensor [1])))))
  (let [e (thrown #(tf/add (tf/tensor [1 2 3]) (tf/tensor [1 2])))]
    (is (= :shape-error (util/error-code e)) "incompatible shapes do not broadcast")
    (is (re-find #"broadcast" (:message (util/error-data e))))))

(deftest elementwise-math-and-activations
  (is (= [1] (tf/->vec (tf/exp (tf/tensor [0])))))
  (is (= [0] (tf/->vec (tf/log (tf/tensor [1])))))
  (is (approx= [1] (tf/->vec (tf/log1p (tf/tensor [(- js/Math.E 1)])))))
  (is (approx= [(- js/Math.E 1)] (tf/->vec (tf/expm1 (tf/tensor [1])))))
  (is (approx= [0 1] (tf/->vec (tf/sin (tf/tensor [0 (/ js/Math.PI 2)])))))
  (is (approx= [1] (tf/->vec (tf/cos (tf/tensor [0])))))
  (is (approx= [0] (tf/->vec (tf/tan (tf/tensor [0])))))
  (is (approx= [0] (tf/->vec (tf/add-n [(tf/asin (tf/tensor [0])) (tf/atan (tf/tensor [0])) (tf/sinh (tf/tensor [0]))
                                        (tf/tanh (tf/tensor [0])) (tf/asinh (tf/tensor [0])) (tf/atanh (tf/tensor [0]))
                                        (tf/erf (tf/tensor [0]))]))))
  (is (approx= [(/ js/Math.PI 2)] (tf/->vec (tf/acos (tf/tensor [0])))))
  (is (approx= [1] (tf/->vec (tf/cosh (tf/tensor [0])))))
  (is (approx= [0] (tf/->vec (tf/acosh (tf/tensor [1])))))
  (is (= [true false] (tf/->vec (tf/is-nan (tf/tensor [##NaN 1])))))
  (is (= [true false] (tf/->vec (tf/is-inf (tf/tensor [##Inf 1])))))
  (is (= [false true] (tf/->vec (tf/is-finite (tf/tensor [##Inf 1])))))
  (is (= [0 1] (tf/->vec (tf/step (tf/tensor [-1 2])))))
  (is (= [1] (tf/->vec (tf/bitwise-and (tf/tensor [3] :int32) (tf/tensor [5] :int32)))))
  (is (= [0 2] (tf/->vec (tf/relu (tf/tensor [-1 2])))))
  (is (= [0 6] (tf/->vec (tf/relu6 (tf/tensor [-1 10])))))
  (is (approx= [-0.1 2] (tf/->vec (tf/leaky-relu (tf/tensor [-1 2]) 0.1))))
  (is (approx= [-0.5 2] (tf/->vec (tf/prelu (tf/tensor [-1 2]) (tf/tensor [0.5])))))
  (is (approx= [0.5] (tf/->vec (tf/sigmoid (tf/tensor [0])))))
  (is (approx= [(js/Math.log 0.5)] (tf/->vec (tf/log-sigmoid (tf/tensor [0])))))
  (is (approx= [(js/Math.log 2)] (tf/->vec (tf/softplus (tf/tensor [0])))))
  (is (approx= [0.5 0.5] (tf/->vec (tf/softmax (tf/tensor [1 1])))))
  (is (approx= [[0.5 0.5] [0.5 0.5]] (tf/->vec (tf/softmax (tf/tensor [[1 1] [2 2]]) 1))))
  (is (approx= [(js/Math.log 0.5) (js/Math.log 0.5)] (tf/->vec (tf/log-softmax (tf/tensor [1 1])))))
  (is (= [1 2] (tf/shape (tf/elu (tf/tensor [[-1 1]])))))
  (is (approx= [0 1.0507] (tf/->vec (tf/selu (tf/tensor [0 1]))) 1e-3))
  (is (= [4] (tf/shape (tf/dropout (tf/ones [4]) 0.5 {:seed 1})))))

(deftest matmul-dot-and-linalg
  (let [a (tf/tensor [[1 2] [3 4]]) b (tf/tensor [[5 6] [7 8]])]
    (is (= [[19 22] [43 50]] (tf/->vec (tf/mat-mul a b))) "hand-computed 2x2 product")
    (is (= [[19 22] [43 50]] (tf/->vec (tf/matmul a b))) "matmul alias")
    (is (= [[26 30] [38 44]] (tf/->vec (tf/mat-mul a b {:transpose-a true}))))
    (is (= [[17 23] [39 53]] (tf/->vec (tf/mat-mul a b {:transpose-b true}))))
    (is (= [[19 22] [43 50]] (tf/->vec (tf/einsum "ij,jk->ik" a b))))
    (is (= [[19 22] [43 50]] (tf/->vec (tf/dot a b)))))
  (is (= 32 (tf/->vec (tf/dot (tf/tensor [1 2 3]) (tf/tensor [4 5 6])))))
  (is (= [[3 4] [6 8]] (tf/->vec (tf/outer-product (tf/tensor [1 2]) (tf/tensor [3 4])))))
  (let [e (thrown #(tf/mat-mul (tf/ones [2 2]) (tf/ones [3 3])))]
    (is (= :shape-error (util/error-code e)))
    (is (= "Error" (:name (util/error-data e))) "tfjs-core asserts throw plain Error"))
  (let [[q r] (tf/linalg-qr (tf/tensor [[1 0] [0 1]]))]
    (is (tf/tensor? q))
    (is (= [2 2] (tf/shape r))))
  (is (= [[1 2] [0 4]] (tf/->vec (tf/linalg-band-part (tf/tensor [[1 2] [3 4]]) 0 -1))))
  (is (= [2 2] (tf/shape (tf/linalg-gram-schmidt (tf/tensor [[1 0] [1 1]])))))
  (is (approx= [[-1 1]] (tf/->vec (tf/batch-norm (tf/tensor [[0 2]]) (tf/tensor [1]) (tf/tensor [1]) {:variance-epsilon 0})) 1e-4))
  (is (= [[2 3]] (tf/->vec (tf/fused-mat-mul {:a (tf/tensor [[1 0]]) :b (tf/tensor [[1 2] [3 4]]) :bias (tf/tensor [1 1]) :activation :relu})))))

(deftest reshape-transpose-and-geometry
  (let [t (tf/tensor [1 2 3 4 5 6])]
    (is (= [[1 2 3] [4 5 6]] (tf/->vec (tf/reshape t [2 3]))))
    (is (= [3 2] (tf/shape (tf/reshape t [-1 2]))))
    (is (= :shape-error (util/error-code (thrown #(tf/reshape t [4 2]))))))
  (is (= [[1 3] [2 4]] (tf/->vec (tf/transpose (tf/tensor [[1 2] [3 4]])))))
  (is (= [2 1 3] (tf/shape (tf/transpose (tf/zeros [1 2 3]) [1 0 2]))))
  (is (= [1 3] (tf/shape (tf/expand-dims (tf/tensor [1 2 3])))))
  (is (= [3 1] (tf/shape (tf/expand-dims (tf/tensor [1 2 3]) 1))))
  (is (= [3] (tf/shape (tf/squeeze (tf/zeros [1 3 1])))))
  (is (= [1 3] (tf/shape (tf/squeeze (tf/zeros [1 3 1]) [2]))))
  (is (= [[1 2] [3 4]] (tf/->vec (tf/concat [(tf/tensor [[1 2]]) (tf/tensor [[3 4]])]))))
  (is (= [[1 2 3 4]] (tf/->vec (tf/concat [(tf/tensor [[1 2]]) (tf/tensor [[3 4]])] 1))))
  (is (= [[1 2] [3 4]] (tf/->vec (tf/stack [(tf/tensor [1 2]) (tf/tensor [3 4])]))))
  (is (= [[1 3] [2 4]] (tf/->vec (tf/stack [(tf/tensor [1 2]) (tf/tensor [3 4])] 1))))
  (let [parts (tf/unstack (tf/tensor [[1 2] [3 4]]))]
    (is (vector? parts))
    (is (= [[1 2] [3 4]] (mapv tf/->vec parts))))
  (is (= [[1 2] [3 4]] (mapv tf/->vec (tf/split (tf/tensor [1 2 3 4]) 2))))
  (is (= [[1] [2 3 4]] (mapv tf/->vec (tf/split (tf/tensor [1 2 3 4]) [1 3]))))
  (is (= [2 3] (tf/->vec (tf/slice (tf/tensor [1 2 3 4]) 1 2))))
  (is (= [[4]] (tf/->vec (tf/slice (tf/tensor [[1 2] [3 4]]) [1 1] [1 1]))))
  (is (= [3 4] (tf/->vec (tf/slice (tf/tensor [1 2 3 4]) 2))))
  (is (= [2 3] (tf/->vec (tf/strided-slice (tf/tensor [1 2 3 4]) [1] [3]))))
  (is (= [1 3] (tf/->vec (tf/strided-slice (tf/tensor [1 2 3 4]) [0] [4] [2]))))
  (is (= [30 10] (tf/->vec (tf/gather (tf/tensor [10 20 30]) (tf/tensor [2 0] :int32)))))
  (is (= [[2] [4]] (tf/->vec (tf/gather (tf/tensor [[1 2] [3 4]]) (tf/tensor [1] :int32) 1))))
  (is (= [4 1] (tf/->vec (tf/gather-nd (tf/tensor [[1 2] [3 4]]) (tf/tensor [[1 1] [0 0]] :int32)))))
  (is (= [0 9 0 8] (tf/->vec (tf/scatter-nd (tf/tensor [[1] [3]] :int32) (tf/tensor [9 8]) [4]))))
  (is (= [1 7 3] (tf/->vec (tf/tensor-scatter-update (tf/tensor [1 2 3]) (tf/tensor [[1]] :int32) (tf/tensor [7])))))
  (is (= [3 2 1] (tf/->vec (tf/reverse (tf/tensor [1 2 3])))))
  (is (= [[2 1] [4 3]] (tf/->vec (tf/reverse (tf/tensor [[1 2] [3 4]]) 1))))
  (is (= [1 2 1 2] (tf/->vec (tf/tile (tf/tensor [1 2]) [2]))))
  (is (= [0 1 2 0] (tf/->vec (tf/pad (tf/tensor [1 2]) [[1 1]]))))
  (is (= [9 1 2] (tf/->vec (tf/pad (tf/tensor [1 2]) [[1 0]] 9))))
  (is (= [2 1 2 1] (tf/->vec (tf/mirror-pad (tf/tensor [1 2]) [[1 1]] :reflect))))
  (is (= [[1 2] [1 2]] (tf/->vec (tf/broadcast-to (tf/tensor [1 2]) [2 2]))))
  (is (= [2 2] (tf/->vec (tf/broadcast-args (tf/tensor [2 1] :int32) (tf/tensor [1 2] :int32)))))
  (is (= [[1 0] [0 2]] (tf/->vec (tf/diag (tf/tensor [1 2])))))
  (is (= [1 2] (tf/shape (tf/ensure-shape (tf/tensor [[1 2]]) [nil 2]))))
  (is (= :shape-error (util/error-code (thrown #(tf/ensure-shape (tf/tensor [[1 2]]) [1 3])))))
  (let [[x y] (tf/meshgrid (tf/tensor [1 2]) (tf/tensor [3 4]))]
    (is (= [[1 2] [1 2]] (tf/->vec x)))
    (is (= [[3 3] [4 4]] (tf/->vec y))))
  (is (= [1 2 2 1] (tf/shape (tf/depth-to-space (tf/zeros [1 1 1 4]) 2 {:data-format :nhwc}))))
  (is (= [1 2 2 1] (tf/shape (tf/depth-to-space (tf/zeros [1 1 1 4]) 2))))
  (let [e (thrown #(tf/depth-to-space (tf/zeros [1 4 1 1]) 2 {:data-format :nchw}))]
    (is (re-find #"NCHW" (:message (util/error-data e))) "op data formats are upper-cased (:nchw -> \"NCHW\"; CPU supports NHWC only)")))

(deftest reduction-ops
  (let [m (tf/tensor [[1 2] [3 4]])]
    (is (= 10 (tf/->vec (tf/sum m))))
    (is (= [4 6] (tf/->vec (tf/sum m 0))))
    (is (= [3 7] (tf/->vec (tf/sum m 1))))
    (is (= [1 2] (tf/shape (tf/sum m 0 true))) "keep-dims")
    (is (= 2.5 (tf/->vec (tf/mean m))))
    (is (= [2 3] (tf/->vec (tf/mean m 0))))
    (is (= 24 (tf/->vec (tf/prod m))))
    (is (= 4 (tf/->vec (tf/max m))))
    (is (= 1 (tf/->vec (tf/min m))))
    (is (= [1 1] (tf/->vec (tf/arg-max m 1))))
    (is (= [0 0] (tf/->vec (tf/arg-min m 1))))
    (is (approx= 5.477226 (tf/->vec (tf/euclidean-norm m)) 1e-5))
    (is (approx= 5.477226 (tf/->vec (tf/norm m)) 1e-5))
    (is (approx= [3 7] (tf/->vec (tf/norm m 1 1))))
    (let [{:keys [mean variance]} (tf/moments m)]
      (is (= 2.5 (tf/->vec mean)))
      (is (= 1.25 (tf/->vec variance)))))
  (let [am (tf/arg-max (tf/tensor [1 3 2]))]
    (is (= 1 (tf/->vec am)))
    (is (= :int32 (tf/dtype am))))
  (is (= [1 3 6] (tf/->vec (tf/cumsum (tf/tensor [1 2 3])))))
  (is (= [0 1 3] (tf/->vec (tf/cumsum (tf/tensor [1 2 3]) 0 {:exclusive true}))))
  (is (= [1 2 6] (tf/->vec (tf/cumprod (tf/tensor [1 2 3])))))
  (is (approx= (js/Math.log 2) (tf/->vec (tf/log-sum-exp (tf/tensor [0 0])))))
  (is (= false (tf/->vec (tf/all (tf/tensor [true false])))))
  (is (= true (tf/->vec (tf/any (tf/tensor [true false])))))
  (is (= [true false] (tf/->vec (tf/all (tf/tensor [[true true] [true false]]) 1)))))

(deftest comparison-logical-and-where
  (let [a (tf/tensor [1 2 3]) b (tf/tensor [1 5 2])]
    (is (= [true false false] (tf/->vec (tf/equal a b))) "bool 1/0 -> true/false at the boundary")
    (is (= :bool (tf/dtype (tf/equal a b))))
    (is (= [false true true] (tf/->vec (tf/not-equal a b))))
    (is (= [false false true] (tf/->vec (tf/greater a b))))
    (is (= [true false true] (tf/->vec (tf/greater-equal a b))))
    (is (= [false true false] (tf/->vec (tf/less a b))))
    (is (= [true true false] (tf/->vec (tf/less-equal a b))))
    (is (= [false true true] (tf/->vec (tf/greater a 1))) "scalar broadcast")
    (is (= [1 5 3] (tf/->vec (tf/where (tf/greater b a) b a)))))
  (let [t (tf/tensor [true false]) f (tf/tensor [true true])]
    (is (= [true false] (tf/->vec (tf/logical-and t f))))
    (is (= [true true] (tf/->vec (tf/logical-or t f))))
    (is (= [false true] (tf/->vec (tf/logical-not t))))
    (is (= [false true] (tf/->vec (tf/logical-xor t f))))))

(deftest async-ops-return-promises
  (async done
    (tu/finish
     (p/let [coords (tf/where! (tf/tensor [[false true] [true false]]))
             masked (tf/boolean-mask! (tf/tensor [1 2 3]) (tf/tensor [true false true]))
             [vals idx] (tf/setdiff1d! (tf/tensor [1 2 3 4]) (tf/tensor [2 4]))
             top (tf/in-top-k! (tf/tensor [[0.1 0.9] [0.9 0.1]]) (tf/tensor [1 1] :int32))
             nms (tf/image-non-max-suppression! (tf/tensor [[0 0 1 1] [0 0 1 1]]) (tf/tensor [0.9 0.8]) 2)]
       (is (= [[0 1] [1 0]] (tf/->vec coords)))
       (is (= [1 3] (tf/->vec masked)))
       (is (= [1 3] (tf/->vec vals)))
       (is (= [0 2] (tf/->vec idx)))
       (is (= [true false] (tf/->vec top)))
       (is (= [0] (tf/->vec nms))))
     done)))

(deftest search-sort-and-sets
  (let [{:keys [values indices]} (tf/topk (tf/tensor [1 3 2]) 2)]
    (is (= [3 2] (tf/->vec values)))
    (is (= [1 2] (tf/->vec indices))))
  (let [{:keys [values indices]} (tf/unique (tf/tensor [1 1 2]))]
    (is (= [1 2] (tf/->vec values)))
    (is (= [0 0 1] (tf/->vec indices))))
  (is (= [[1 2]] (tf/->vec (tf/search-sorted (tf/tensor [[1 3 5]]) (tf/tensor [[2 4]])))))
  (is (= [[0]] (tf/->vec (tf/lower-bound (tf/tensor [[1 3]]) (tf/tensor [[1]])))))
  (is (= [[1]] (tf/->vec (tf/upper-bound (tf/tensor [[1 3]]) (tf/tensor [[1]])))))
  (is (= [1 2 0] (tf/->vec (tf/bincount (tf/tensor [0 1 1] :int32) [] 3))))
  (is (= [[1 1]] (tf/->vec (tf/dense-bincount (tf/tensor [[0 1]] :int32) [] 2))))
  (is (= [[4 6]] (tf/->vec (tf/unsorted-segment-sum (tf/tensor [[1 2] [3 4]]) (tf/tensor [0 0] :int32) 1))))
  (is (= [0 5 0] (tf/->vec (tf/sparse-to-dense (tf/tensor [1] :int32) (tf/tensor [5]) [3])))))

(deftest data-access-copies-out
  (let [t (tf/tensor [[1 2 3] [4 5 6]])]
    (is (= [1 2 3 4 5 6] (tf/data-sync t)) "flat row-major copy")
    (is (= 6 (tf/get-element t 1 2)))
    (is (= [3] (tf/strides t)) "tfjs omits the trailing stride of 1")
    (is (= [12 4] (tf/strides (tf/zeros [2 3 4]))))
    (is (re-find #"Tensor" (tf/->string t)))
    (is (re-find #"dtype" (tf/->string t {:verbose true}))))
  (is (= [true false] (tf/data-sync (tf/tensor [true false]))) "bool storage 1/0 -> booleans")
  (is (= true (tf/get-element (tf/tensor [true false]) 0)))
  (is (not (tf/disposed? (tf/scalar 1))))
  (async done
    (tu/finish
     (p/let [v (tf/->vec! (tf/tensor [[1 2]]))
             d (tf/data! (tf/tensor [true false]))
             b (tf/bytes! (tf/tensor [1]))
             sb (tf/bytes! (tf/tensor ["ab"]))]
       (is (= [[1 2]] v))
       (is (= [true false] d))
       (is (= [0 0 128 63] b) "float32 1.0 little-endian")
       (is (= [[97 98]] sb) "UTF-8 bytes per string element"))
     done)))

(deftest string-and-complex-tensors
  (let [s (tf/tensor ["a" "bc"])]
    (is (= ["a" "bc"] (tf/->vec s)))
    (is (= ["a" "bc"] (tf/data-sync s)))
    (is (= :shape-error (util/error-code (thrown #(tf/add s s)))) "string tensors are rejected by numeric ops (dtype demand -> :shape-error)"))
  (let [{:keys [indices values shape]} (tf/string-split (tf/tensor ["a b" "c"]) " ")]
    (is (= ["a" "b" "c"] (tf/->vec values)))
    (is (= [[0 0] [0 1] [1 0]] (tf/->vec indices)))
    (is (= [2 2] (tf/->vec shape))))
  (is (= ["x-y"] (tf/->vec (tf/static-regex-replace (tf/tensor ["x y"]) " " "-"))))
  (is (= :int32 (tf/dtype (tf/string-to-hash-bucket-fast (tf/tensor ["a"]) 10))))
  (let [c (tf/complex [1 2] [3 4])]
    (is (= :complex64 (tf/dtype c)))
    (is (= [2] (tf/shape c)))
    (is (= [1 3 2 4] (tf/->vec c)) "interleaved [re im re im]")
    (is (= [1 2] (tf/->vec (tf/real c))))
    (is (= [3 4] (tf/->vec (tf/imag c))))
    (is (= [2] (tf/shape (tf/fft c))))
    (is (approx= [1 2] (tf/->vec (tf/real (tf/ifft (tf/fft c)))) 1e-5)))
  (is (= [3] (tf/shape (tf/rfft (tf/tensor [1 2 3 4]))))))

(deftest tidy-keeps-only-the-returned-container
  (let [before (tf/num-tensors)
        r (tf/tidy (fn []
                     (let [a (tf/tensor [1 2])
                           b (tf/add a 1)
                           c (tf/mul b 2)
                           _leak (tf/sub c 1)]
                       {:sum (tf/sum c) :parts [a c] :label "x"})))]
    (is (= 3 (- (tf/num-tensors) before)) "3 tensors reachable from the CLJS map survive, 2 intermediates are freed")
    (is (= 10 (tf/->vec (:sum r))) "sum of ((a+1)*2) = 4+6")
    (is (= [4 6] (tf/->vec (second (:parts r)))))
    (is (= "x" (:label r)))
    (tf/dispose! r)
    (is (= before (tf/num-tensors)) "dispose! walks the CLJS container")
    (is (tf/disposed? (:sum r))))
  (let [before (tf/num-tensors)
        t (tf/tidy "named" (fn [] (tf/tensor [1])))]
    (is (= (inc before) (tf/num-tensors)))
    (tf/dispose! t)
    (is (= before (tf/num-tensors))))
  (let [before (tf/num-tensors)
        kept (volatile! nil)]
    (tf/tidy (fn [] (vreset! kept (tf/keep (tf/tensor [1]))) (tf/tensor [2]) nil))
    (is (= (inc before) (tf/num-tensors)) "keep exempts a tensor from the scope")
    (tf/dispose! @kept))
  (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(tf/tidy 3)))))
  (let [t (tf/tensor [1])]
    (tf/dispose! t)
    (is (= :tensorflow-cljs/tf-error (util/error-type (thrown #(tf/->vec t))))))
  (testing "uncertainty 11.4: tidy over a large persistent structure is cheap"
    (let [before (tf/num-tensors)
          t0 (js/Date.now)
          r (tf/tidy (fn [] {:many (into {} (map (fn [i] [i (tf/scalar i)])) (cljs.core/range 200))}))
          ms (- (js/Date.now) t0)]
      (is (= 200 (- (tf/num-tensors) before)))
      (is (< ms 250) (str "200-tensor map through tidy took " ms " ms"))
      (tf/dispose! r)
      (is (= before (tf/num-tensors)))))
  (let [m (tf/memory)]
    (is (number? (:num-tensors m)))
    (is (number? (:num-bytes m)))
    (is (contains? m :unreliable))))

(deftest variables-gradients-and-minimize
  (let [v (tf/variable (tf/tensor [1 2]))]
    (is (tf/variable? v))
    (is (tf/tensor? v))
    (tf/assign! v [3 4])
    (is (= [3 4] (tf/->vec v)))
    (is (= :shape-error (util/error-code (thrown #(tf/assign! v [1 2 3])))))
    (tf/dispose! v))
  (let [w (tf/variable (tf/scalar 0))
        opt (tf/make-optimizer "sgd" [0.1])
        cost (tf/minimize! opt (fn [] (tf/square (tf/sub w 1))))]
    (is (tf/optimizer? opt))
    (is (= "SGD" (tf/optimizer-class-name opt)))
    (is (= 1 cost) "cost before the step: (0-1)^2")
    (is (approx= 0.2 (tf/->vec w)) "w <- w - 0.1 * 2(w-1)")
    (let [{:keys [value grads]} (tf/variable-grads (fn [] (tf/square w)))]
      (is (approx= 0.04 (tf/->vec value)))
      (is (= 1 (count grads)))
      (is (approx= 0.4 (tf/->vec (first (vals grads))))))
    (let [{:keys [value grads]} (tf/compute-gradients opt (fn [] (tf/mul w 3)))]
      (is (approx= 0.6 (tf/->vec value)))
      (is (approx= 3 (tf/->vec (first (vals grads)))))
      (tf/apply-gradients! opt grads)
      (is (approx= -0.1 (tf/->vec w)) "w <- 0.2 - 0.1*3"))
    (tf/dispose! w))
  (is (= [6] (tf/->vec ((tf/grad (fn [x] (tf/square x))) (tf/tensor [3])))))
  (let [{:keys [value grad]} ((tf/value-and-grad (fn [x] (tf/sum (tf/square x)))) (tf/tensor [1 2]))]
    (is (= 5 (tf/->vec value)))
    (is (= [2 4] (tf/->vec grad))))
  (is (= [[2] [3]] (mapv tf/->vec ((tf/grads (fn [[a b]] (tf/mul a b))) [(tf/tensor [3]) (tf/tensor [2])]))))
  (let [twice (tf/custom-grad (fn [[x]] {:value (tf/mul x 2) :grad-func (fn [dy] [(tf/mul dy 10)])}))]
    (is (= [4] (tf/->vec (twice (tf/tensor [2])))))
    (is (= [10] (tf/->vec ((tf/grad (fn [x] (twice x))) (tf/tensor [2]))))))
  (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(tf/make-optimizer "bogus" []))))))

(deftest convolution-and-pooling
  (let [img (tf/ones [1 3 3 1])
        k (tf/ones [2 2 1 1])
        out (tf/conv2d img k)]
    (is (= [1 2 2 1] (tf/shape out)))
    (is (= [[[[4] [4]] [[4] [4]]]] (tf/->vec out)) "2x2 window of ones sums to 4")
    (is (= [1 3 3 1] (tf/shape (tf/conv2d img k {:pad :same}))))
    (is (= [1 1 1 1] (tf/shape (tf/conv2d img k {:strides 2}))))
    (is (= [1 3 3 1] (tf/shape (tf/conv2d-transpose out k [1 3 3 1]))))
    (is (= [1 2 2 1] (tf/shape (tf/depthwise-conv2d img k))))
    (is (= [1 2 2 1] (tf/shape (tf/separable-conv2d img k (tf/ones [1 1 1 1])))))
    (is (= [1 2 2 1] (tf/shape (tf/dilation2d img (tf/ones [2 2 1])))))
    (is (= [[[[4] [4]] [[4] [4]]]] (tf/->vec (tf/fused-conv2d {:x img :filter k :strides 1 :pad :valid})))))
  (is (= [1 2 1] (tf/shape (tf/conv1d (tf/ones [1 3 1]) (tf/ones [2 1 1])))))
  (is (= [1 1 1 1 1] (tf/shape (tf/conv3d (tf/ones [1 2 2 2 1]) (tf/ones [2 2 2 1 1])))))
  (let [m (tf/tensor [[[[1] [2]] [[3] [4]]]])]
    (is (= [[[[4]]]] (tf/->vec (tf/max-pool m 2))))
    (is (= [[[[2.5]]]] (tf/->vec (tf/avg-pool m 2))))
    (is (= [1 2 2 1] (tf/shape (tf/max-pool m 2 {:strides 1 :pad :same}))))
    (is (= [[[[4]]]] (tf/->vec (tf/pool m 2 :max))))
    (let [{:keys [result indexes]} (tf/max-pool-with-argmax m 2)]
      (is (= [[[[4]]]] (tf/->vec result)))
      (is (= [[[[3]]]] (tf/->vec indexes)))))
  (is (= [1 1 1 1 1] (tf/shape (tf/max-pool3d (tf/ones [1 2 2 2 1]) 2))))
  (is (= [1 1 1 1 1] (tf/shape (tf/avg-pool3d (tf/ones [1 2 2 2 1]) 2))))
  (is (= [1 1 1 2] (tf/shape (tf/local-response-normalization (tf/ones [1 1 1 2])))))
  (is (= [4 2 2 1] (tf/shape (tf/space-to-batch-nd (tf/ones [1 4 4 1]) [2 2] [[0 0] [0 0]]))))
  (is (= [1 4 4 1] (tf/shape (tf/batch-to-space-nd (tf/ones [4 2 2 1]) [2 2] [[0 0] [0 0]]))))
  (let [[c h] (tf/basic-lstm-cell (tf/scalar 1) (tf/zeros [2 4]) (tf/zeros [4]) (tf/zeros [1 1]) (tf/zeros [1 1]) (tf/zeros [1 1]))]
    (is (= [1 1] (tf/shape c)))
    (is (= [1 1] (tf/shape h)))))

(deftest image-signal-and-misc
  (is (= [1 2 2 1] (tf/shape (tf/image-resize-nearest-neighbor (tf/ones [1 1 1 1]) [2 2]))))
  (is (= [[[[1] [1]] [[1] [1]]]] (tf/->vec (tf/image-resize-bilinear (tf/ones [1 1 1 1]) [2 2]))))
  (is (= [[[[2] [1]]]] (tf/->vec (tf/image-flip-left-right (tf/tensor [[[[1] [2]]]])))))
  (is (= [1 1 1 3] (tf/shape (tf/image-grayscale-to-rgb (tf/ones [1 1 1 1])))))
  (is (= [1 1 1 1] (tf/shape (tf/image-rgb-to-grayscale (tf/ones [1 1 1 3])))))
  (is (= [1 2 2 1] (tf/shape (tf/image-rotate-with-offset (tf/ones [1 2 2 1]) 1.0))))
  (is (= [1 2 2 1] (tf/shape (tf/image-crop-and-resize (tf/ones [1 2 2 1]) (tf/tensor [[0 0 1 1]]) (tf/tensor [0] :int32) [2 2]))))
  (is (= [0] (tf/->vec (tf/image-non-max-suppression (tf/tensor [[0 0 1 1] [0 0 1 1]]) (tf/tensor [0.9 0.8]) 2))))
  (let [{:keys [selected-indices selected-scores]} (tf/image-non-max-suppression-with-score (tf/tensor [[0 0 1 1]]) (tf/tensor [0.9]) 1)]
    (is (= [0] (tf/->vec selected-indices)))
    (is (approx= [0.9] (tf/->vec selected-scores))))
  (let [{:keys [selected-indices valid-outputs]} (tf/image-non-max-suppression-padded (tf/tensor [[0 0 1 1] [0 0 1 1]]) (tf/tensor [0.9 0.8]) 2 {:pad-to-max-output-size true})]
    (is (= [2] (tf/shape selected-indices)) "padded to max-output-size")
    (is (= 1 (tf/->vec valid-outputs))))
  (is (= [2 2 1] (tf/shape (tf/image-threshold (tf/ones [2 2 1])))))
  (is (= [1 2 2 1] (tf/shape (tf/image-transform (tf/ones [1 2 2 1]) (tf/tensor [[1 0 0 0 1 0 0 0]])))))
  (is (= [4] (tf/shape (tf/signal-hann-window 4))))
  (is (= [4] (tf/shape (tf/signal-hamming-window 4))))
  (is (= [2 2] (tf/shape (tf/signal-frame (tf/tensor [1 2 3 4]) 2 2))))
  (is (= [1 3] (tf/shape (tf/signal-stft (tf/tensor [1 2 3 4]) 4 1))))
  (is (= 8 (tf/enclosing-power-of-two 5)))
  (is (= [3] (tf/shape (tf/cosine-window 3 0.5 0.5))))
  (is (approx= [1.9] (tf/->vec (tf/moving-average (tf/tensor [1]) (tf/tensor [10]) 0.9 {:zero-debias false}))))
  (let [{:keys [rt-nested-splits rt-dense-values]} (tf/ragged-range (tf/tensor [0] :int32) (tf/tensor [3] :int32) (tf/tensor [1] :int32))]
    (is (= [0 3] (tf/->vec rt-nested-splits)))
    (is (= [0 1 2] (tf/->vec rt-dense-values))))
  (is (= [[1 2]] (tf/->vec (tf/sparse-segment-sum (tf/tensor [[1 2] [3 4]]) (tf/tensor [0] :int32) (tf/tensor [0] :int32)))))
  (is (= [[2 3]] (tf/->vec (tf/sparse-segment-mean (tf/tensor [[1 2] [3 4]]) (tf/tensor [0 1] :int32) (tf/tensor [0 0] :int32))))))

(deftest losses-and-metric-functions
  (is (= 2 (tf/->vec (tf/losses-mean-squared-error (tf/tensor [1 2]) (tf/tensor [1 4])))))
  (is (= [0 4] (tf/->vec (tf/losses-mean-squared-error (tf/tensor [1 2]) (tf/tensor [1 4]) {:reduction :none}))))
  (is (= 4 (tf/->vec (tf/losses-mean-squared-error (tf/tensor [1 2]) (tf/tensor [1 4]) {:reduction :sum}))))
  (is (= 1 (tf/->vec (tf/losses-absolute-difference (tf/tensor [1 2]) (tf/tensor [1 4])))))
  (is (approx= 1.5 (tf/->vec (tf/losses-huber-loss (tf/tensor [0]) (tf/tensor [2])))) "delta (|e| - delta/2) = 1 * 1.5")
  (is (approx= 0 (tf/->vec (tf/losses-hinge-loss (tf/tensor [1]) (tf/tensor [1])))))
  (is (approx= 0 (tf/->vec (tf/losses-log-loss (tf/tensor [1]) (tf/tensor [1]))) 1e-4))
  (is (approx= 0 (tf/->vec (tf/losses-cosine-distance (tf/tensor [[1 0]]) (tf/tensor [[1 0]]) 1))))
  (is (approx= (js/Math.log 2) (tf/->vec (tf/losses-softmax-cross-entropy (tf/tensor [[1 0]]) (tf/tensor [[0 0]])))))
  (is (approx= (js/Math.log 2) (tf/->vec (tf/losses-sigmoid-cross-entropy (tf/tensor [[1]]) (tf/tensor [[0]])))))
  (is (= 2 (tf/->vec (tf/losses-compute-weighted-loss (tf/tensor [1 3])))))
  (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(tf/losses-mean-squared-error (tf/tensor [1]) (tf/tensor [1]) {:reduction :bogus})))))
  (is (= [1] (tf/->vec ((tf/metric-fn "meanAbsoluteError") (tf/tensor [[1 2]]) (tf/tensor [[2 3]])))))
  (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(tf/metric-fn "bogus"))))))

(deftest buffers
  (let [b (tf/buffer [2 2])]
    (tf/buffer-set! b 5 0 1)
    (is (= 5 (tf/buffer-get b 0 1)))
    (is (= [[0 5] [0 0]] (tf/->vec (tf/buffer->tensor b)))))
  (is (= [[1 2]] (tf/->vec (tf/buffer->tensor (tf/buffer [1 2] {:values [1 2]})))))
  (is (= 4 (tf/buffer-get (tf/->buffer (tf/tensor [[1 2] [3 4]])) 1 1))))

(deftest errors-backend-and-environment
  (let [e (thrown #(tf/mat-mul (tf/ones [2 2]) (tf/ones [1 3])))]
    (is (= :tensorflow-cljs/tf-error (util/error-type e)))
    (is (util/shape-error? e))
    (is (some? (:cause (util/error-data e))))
    (is (re-find #"^tensorflow-cljs shape-error: Error in matMul" (ex-message e))))
  (is (= true (tf/env-flag :is-node)))
  (is (= true (tf/env-flag "IS_NODE")))
  (is (contains? (tf/env-flags) :is-node))
  (is (tf/find-backend "cpu"))
  (is (not (tf/find-backend "webgl")))
  (is (= "4.22.0" (:tfjs (tf/version))))
  (async done
    (tu/finish
     (p/let [_ (tf/ready!)
             ok (tf/set-backend! "cpu")
             bad (tu/rejection (tf/set-backend! "webgl"))
             timing (tf/time! (fn [] (tf/mat-mul (tf/ones [4 4]) (tf/ones [4 4]))))
             prof (tf/profile! (fn [] (tf/add (tf/ones [2]) 1)))]
       (is (= "cpu" (tf/backend)))
       (is (true? ok))
       (is (= :tensorflow-cljs/backend-error (util/error-type bad)))
       (is (= "webgl" (:backend (ex-data bad))))
       (is (number? (:wall-ms timing)))
       (is (= 2 (:new-tensors prof)) "the ones and the sum")
       (is (tf/tensor? (:result prof))))
     done)))

(deftest dtype-promotion-and-typed-arrays
  (is (= :float32 (tf/dtype (tf/mul (tf/tensor [1] :int32) (tf/tensor [0.5])))))
  (is (= :int32 (tf/dtype (tf/add (tf/tensor [1] :int32) (tf/tensor [2] :int32)))))
  (is (= :bool (tf/dtype (tf/logical-and (tf/tensor [true]) (tf/tensor [false])))))
  (is (= [1 0 3] (tf/->vec (tf/cast (tf/tensor [1.9 0.2 3.7]) :int32))) "float -> int32 truncates")
  (is (= [true false] (tf/->vec (tf/cast (tf/tensor [2 0]) :bool))))
  (is (= :float32 (tf/dtype (tf/tensor (js/Float32Array. #js [1 2])))) "typed arrays are flat data")
  (is (= :int32 (tf/dtype (tf/tensor (js/Int32Array. #js [1 2])))))
  (is (= [[1 2]] (tf/->vec (tf/tensor (js/Float32Array. #js [1 2]) {:shape [1 2]}))))
  (let [[a b] (tf/->vec (tf/tensor [1 "a"]))]
    (is (= 1 a))
    (is (js/isNaN b) "mixed input: NaN for the string, no error (tfjs)")))

(deftest memory-report-and-counting
  (let [before (tf/num-tensors)
        m0 (tf/memory)
        t (tf/zeros [10])
        m1 (tf/memory)]
    (is (= (inc before) (tf/num-tensors)))
    (is (= 40 (- (:num-bytes m1) (:num-bytes m0))) "10 float32 = 40 bytes")
    (is (= (inc (:num-tensors m0)) (:num-tensors m1)))
    (is (true? (:unreliable m1)) "CPU counts are upper bounds")
    (is (vector? (:reasons m1)))
    (tf/dispose! t)
    (is (= before (tf/num-tensors)))
    (tf/dispose! [t]) 
    (is (= before (tf/num-tensors)) "disposing twice is harmless")))

(deftest higher-rank-creators-and-rendering
  (is (= [1 1 1 1 2] (tf/shape (tf/tensor5d [[[[[1 2]]]]]))))
  (is (= [1 1 1 1 1 2] (tf/shape (tf/tensor6d [[[[[[1 2]]]]]]))))
  (is (= [2 3] (tf/shape (tf/tensor2d [1 2 3 4 5 6] {:shape [2 3]}))))
  (is (= :tensorflow-cljs/argument-error (util/error-type (thrown #(tf/tensor3d [1 2] {:shape [2]})))) "rank of an explicit shape is checked")
  (is (= "Tensor\n    [[1, 2],\n     [3, 4]]" (tf/->string (tf/tensor [[1 2] [3 4]]))))
  (is (= :int32 (tf/dtype (tf/scalar 1 :int32))))
  (is (= "x" (tf/->vec (tf/scalar "x")))))

(deftest broadcasting-across-ranks
  (let [a (tf/ones [2 1 3]) b (tf/ones [4 1])]
    (is (= [2 4 3] (tf/shape (tf/add a b))) "[2 1 3] + [4 1] -> [2 4 3]")
    (is (= [2 4 3] (tf/shape (tf/mul a b))))
    (is (= [2 4 3] (tf/shape (tf/greater a b))))
    (is (= [2 4 3] (tf/shape (tf/where (tf/greater a b) a b)))))
  (is (= [[2 3] [4 5]] (tf/->vec (tf/add (tf/tensor [[1 2] [3 4]]) (tf/scalar 1)))) "scalar tensor broadcasts")
  (is (= [[0 0] [0 0]] (tf/->vec (tf/sub (tf/tensor [[1 2] [3 4]]) [[1 2] [3 4]]))) "nested vectors accepted as operands")
  (is (= :shape-error (util/error-code (thrown #(tf/mul (tf/ones [2 3]) (tf/ones [3 2])))))))
