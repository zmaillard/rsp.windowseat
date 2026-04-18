(ns queue-sign
  (:require [babashka.curl :as curl]
           [cheshire.core :as json])) 



(def runpod-settings {:runpod-key (System/getenv "RUNPOD_API_KEY")
                      :webhook (System/getenv "WEBHOOK")
                      :runpod-worker (System/getenv "RUNPOD_WORKER")})

(defn enqueue 
 [imageid]
 (let [{token :runpod-key webhook :webhook worker :runpod-worker} runpod-settings
       url (str "https://api.runpod.ai/v2/" worker "/run")] 
   (curl/post url
              {:headers {:content-type "application/json"
                         :authorization (str "Bearer " token)}
               :body (json/encode {:webhook webhook :input {:imageid imageid}})})))






(defn -main
  [& [imageid]]
 (enqueue imageid))
