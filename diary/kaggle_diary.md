<div style="color: white;">

# 2020/10/11

* 100Mのデータを処理するのは難しい
  * user_idを10で割ったあまりで、10個のデータに分ける→10個のモデルを作るってのはどうだろう?
  ![image](image.png)

# 2020/10/12
* mlflowを導入
* target_encoding, intelligence_scoreの導入
  * intelligence_score: 問題毎に正解率の逆数を付与(難しいほど高い)
* とりあえずsubmissionスクリプト書いて回してみたけど…通るだろうか…

# 2020/10/13
* pickle -> featherで試してみる
* floatの不要な特徴は全部削除
* target encodingとかの特徴作成に時間かかってる。loadingはfeatherにしたことで解消した。

* 結局、partial_fitをすることで解消(PartialAggregator, ex005_pipeline)

# 2020/10/14
* exp005のモデル提出 CV:0.757(10fold-mean), LB: 0.743
* exp006: groupbyの項目を増やして試す
  * 全面的にクラス設計を見なおした
* exp007: 全データの1%のユーザーの行は全部validationに -> CVそんな変わらず
</div>