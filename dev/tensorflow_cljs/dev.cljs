(ns tensorflow-cljs.dev
  "Node-script entry point for interactive development."
  (:require [tensorflow-cljs.util :as util]))

(defn main [& _]
  (println "tensorflow-cljs dev runtime loaded. Attach with: npx shadow-cljs cljs-repl dev")
  (println "error types:" (sort util/error-types))
  (println "dtypes:" (sort (keys util/dtypes)))
  (js/setInterval #(println "…dev heartbeat") 60000))
