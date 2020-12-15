
* rating
    * EloのRating(ゲームでよく使われる実力の指標)で、content_idとuser_idのRateをつける。
    * content_idのRatingは事前に計算し、辞書として持っておく
    * user_id_rating, content_id_rating, (user_id, part)_rating
    * user_id_rating - content_id_rating
      Elo Ratingは、ratingの差によって勝率が統計的に求められるらしい
      
* userが同じcontent_idを何回前に解いたか?
    * n>=2だと正解率0.9くらいになる。n=1だと正解率は0.6くらい

* 時系列を表現したい特徴
  * 今解く問題について、過去100問のそれぞれのcontent_idで正解した場合・不正解だった場合の正解率 
    -> mean, max, min, sum, std (+0.002)
  * 今解く問題について、過去にあるlectureを受けた場合、受けてない場合の正解率
    -> mean, max, min, sum, std (+0.0005)

* timestamp系
  * 上記の特徴の生値および、user_id, content_id, (user_id, part)で平均とったりその差分とったり
    * timestamp-delta(1~10)
    * timestampを扱うときは、同じtask_container_idだった場合はtask_container_idで割る(+0.0003)
    * timestamp-delta - prior_question_elapsed_time　←なぜか効いてる
    * max(timestamp-delta, 100000)　平均とるときに極端な外れ値に影響されないようにしたいため
    * content_idごとのelapsed_time(※)の平均 ※df.groupby("user_id")["prior_question_elapsed_time"].shift(-1)

  * timestamp-delta - df.groupby("content_id")["elapsed_time"].mean() (+0.0003)
  * df.groupby("content_id", "answered_correctly")["timestamp-delta"].mean()
    -> 正解時平均/不正解時平均とtimestamp-deltaの差をとる

* なんかよくわかんないけど効いてる(+0.0005)
  * part2, part5の実力
      実力の測り方: mean(target_encoding(user_id) - target_encoding(content_id))
  
* その他考察
  * 間違えてelapsed_timeを特徴に入れたとき、CV+0.006だった。解く時間を推定することができればかなりいいのでは
  * tagsの情報を使えてない
  * user_answerの情報を使えてない。。df.groupby(content_id, past_content_id, past_user_answer)["answered_correctly"].mean().to_dict()くらい
* ??
  * tags+partでOneHotEncodingしてK-Meansクラスタリングし、groupby(user_id, cluster_no)["mean"]

* not worked
  * user_answerを当てに行くタスクに変えた　→　木がなかなか収束しないためPDCA回せないと判断し不採用。アンサンブルしたらCV+0.001だったが…
  * 