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

# 2020/10/15
* exp006の1モデル　CV:0.760 LB: 0.739
* exp008: CV: 0.761
  * trainはsplitだけとtestは全部マージしていることによるバグ修正
    * user_id削除
    * countencodingはuser_id以外は10倍にする
  * 項目増やす
* exp009 => CV:0.751
  * question, lectureの追加
  * answered_correctly=-1(lecture)も訓練してたバグ修正
* exp010 => parameter tuning. iteration少ない方が良さそう
  * lr=0.3にしてあといろいろ

# 2020/10/16
* exp009のpredict改良、fitは毎回やらない(10000件ごとに実施)
* exp011: 10model lr=0.3 => CV:0.754 LB:0.747
* exp012: 1model/50Mrow data
  * 固まって動かない。30Mrowでもダメ…辛い…
* exp013: tags1~tags6の特徴追加
  * tags1~tags6はあんまり効果なさそう
  * 20M行にして夜放置してみる

# 2020/10/17
## experiment
* exp013: CV: 0.756(10modelと比べて+0.005) LB: 0.742
* exp013-2: exp013はsplit_numミスってたので、それの修正 これとexp011-2どっちがいいか？
* exp014: 3model(30Mrow) => CV 0.757(1model, 2300epoch)途中で終わり
* exp011-2: exp013と同じモデルで、fitの頻度を10000=>300に変更: LB 0.750
* exp015: TargetEncoding with initial weight
  * user_idとtimestampでソートするよう修正。。
* exp016: TargetEncoding with initial weight + user_ability
##  EDA
  * tags 003_tags_vs_targetencoder
    * ![image_1](image_1.png)
    * まあ、tags1~6のcountencodingで十分カバーできてるかなぁ…
  * user_id 004_user_id_answered_correctly
    * 成績悪い人、成績いい人を比較する
      * 成績悪い人はlecture受けてる件数が多い
        * ![image_2](image_2.png)
      * いい人はlecture受けてる件数が少ない
        * ![image_3](image_3.png)
  * user_level
    * 頭が良ければ正解率高いとは限らん。その人にあった問題を選んで勉強するはず。そのレベルを見る。
      * ![image_4](image_4.png)
      * ![image_5](image_5.png)
    * target_encoding, 初期値ブレがち。200件*0.655を最初から加重しとくのがよさそう！ -> exp015
      * ![image_6](image_6.png)

# 2020/10/18
## experiment
* exp015: TargetEncoding with initial weight => CV: 0.752(1model) +0.002 from exp011 -> LB: 0.732(!?)
  * user_idとtimestampでソートするよう修正。
* exp016: TargetEncoding with initial weight + user_ability => CV: 0.756) +0.006 from exp011 -> LB: 0.666(!?!?)
* exp017: initial_score, initial_weight設定忘れてる...
  * あと、なんかexp015, exp016がuser_idとcontent_idの順番逆なのになぜか動いていたっぽい。けど今は動かない。
  * スコアが低いのもそこらへんでなんかバグってるのが要因では。。
* exp018: exp011を20model -> LB: 0.750
* exp019: exp011 + nunique, shiftdiff
## EDA
* 006_previous_X
  * questionに絞ると、前のquestionと今のquestionが一緒の場合、正解率は高い(当たり前か!?)
  * ![image_7](image_7.png)
  * timestamp_diffみると、1000~10000は正解率低い
  * ![image_8](image_8.png)

# 2020/10/19
* exp019 -> CV: 0.764 / LB: 0.751
* exp020: exp011を丸コピ(exp019が上がりすぎてるので怪しい。。) -> CV戻った
* exp021: user_levelのみ + exp019 (sort_valueなし) -> CV: 0.769 / LB: 0.711 :(
* exp022: exp021 + lecture.csvなし
* exp023: exp021 + type_of, tag 使う

# 2020/10/20
* predictionの特徴作成本当にあってるか、確認するスクリプト作成 ok
* exp023: target_encoderがリークしてた…のか。
  * is_partial_fit：未来のデータで過去のデータをencodeしている
* exp024: environmentちゃんと動いてるか確認(CVとかは見ない) exp021でやる
* exp021_2: exp021 + partial_predict_mode変更

# 2020/10/21
## EDA
* 007_shiftdiff
  * ![image_9](image_9.png)

## experiment
* exp019_2: shiftdiffをリアルタイムに -> LB: 0.756!

# 2020/10/22
## experiment
* exp025: user_levelとnuniqueを削って、content_id系だけ早く更新する -> CV: 0.723?
  * バグってるので要確認
* exp026: ("user_id", "content_id")を加える
* exp027: parameter tuning 旅行中

# 2020/10/23
## experiment
* exp025-2: update_record=50

# 2020/10/25
## experiment
* exp028 parameter tuning結果の反映(base: exp019)

# 2020/10/26
## EDA
* answerの精度が低い人、user_answerが0, 3に固まりがち

![image_10](image_10.png)

* そもそも、answer=2が少ない…

![image_11](image_11.png)

# 2020/10/27
## experiment
* exp029: 各target_encをuser_idのtarget_encとぶつける -> CVかわらず

# 2020/10/28
## experiment
* exp030: user_levelに再トライ -> 1modelのみでいったんsub
* exp031: counter(その時点までに何回該当カテゴリを通っているか?)

# 2020/10/29
* exp032: user_levelだけにしてpartial_predictとall_predict比較
-> ![image_12](image_12.png)
-> LB: 0.766!
* そのまま、exp031と同条件で、partial vs all での精度を比較

# 2020/10/30
## EDA
* 過去にこのユーザーが同じquestionを解いたか
![image_13](image_13.png)

## experiment
* exp033: exp031 + 1個前の解答 (とりあえず生コードで）-> CV: 0.772
* だけどメモリーエラー…

# 2020/10/31
## experiment
* exp034: shiftdiff(content_id)! (直近が同じか?) -> あんまり変わらず
* exp035: exp030と同じ特徴量、CountEncoderのバグ修正したやつ
* exp036
  * content_type_idのtarget_encodingは不要
  * partはtarget_encoding, level必要
  * {prior_question_had_explanation, correct_answer}
  * {prior_question_had_explanation, part}
* exp037
  * tag全部もり(subは流石に無理そうなので無し)
## EDA
* 011_user_answer
  * correct_answerのバランス悪いのなんでだろう
    * ![image_14](image_14.png)
  * exp036に、{prior_question_had_explanation, correct_answer}を加える

* 012_prior_question
  * partとぶつける
    * ![image_16](image_16.png)
  * exp036に、{prior_question_had_explanation, part}を加える

* 013_bundle_id
  * 特になし。。

* 014_part
  * part vs 時間diffで相関あるか？ -> part7は結構顕著に出てる
    * ![image_17](image_17.png)
    * exp036に、groupby("part")["time"].mean()
  * partごとのtarget平均
    * ![image_18](image_18.png)
  * user_id/partごとの正解率相関
    * ![image_19](image_19.png)
    * ![image_20](image_20.png)

# 2020/11/1
## experiment
* ex_036: CV: 0.771 LB: 0.775!!
* ex_038
  * max_bin=1024 -> そんなに変わらない
  * feature_fraction=0.1
* ex_037(tag全部もり) -> あんまりスコア変わらない
* ex_039
  * count_bin追加, partごと/count_binごとのCategoryLevelEncoder
  * diff_rate_mean_target_enc_part_1が強い。part1なんか大事そう
* ex_040
  * prior_question+user_count_bin, content_id+prior_question_had_explanation
  * part+prior_question_elapsed_time_bin

## eda
* 016_user_id
  * ![image_21](image_21.png)
  * prior_question_had_explanationは、最初のほうが顕著に正解率下がる
    * ![image_22](image_22.png)
* 017_prior_question_elapsed_time
  * partごとに、かけた時間と正解率が違う。確かに、文法なんかはわかったら一瞬

# 2020/11/2
## experiment
* ex_039はtimeup
  * CategoryLevelEncoderが時間食ってる(あたりまえ)
* ex_041 -> CV: 0.771 / LB: 0.777
  * CategoryLevel: part=(2, 5), bin=(0)
  * pickle化も同時にやる
* ex_042 -> model1: CV0.768をみて撤退
  * base_ex041
  * LevelEncoder系、全部[content_id, prior_question_had_explanation]にする
* ex_043 ->
  * base_ex_041
  * PriorQuestionElapsedTimeDiv10Encoder -> ほとんどfeatureimporance0

## EDA
* 018_prior_question_time
  * 10で割ったら…
  * ![image_23](image_23.png)

# 2020/11/3
## experiment
* ex_046: hyperparam tuning

# 2020/11/4
## experiment
* ex_047: parameter tuned & lr=0.1 -> CV+0.001だがLBあまりかわらず
* ex_048: previous_answer(reduce memory)
* ex_049: reduce features => CV: 0.776 / LB: 0.784(freq=30)

# 2020/11/6
## experiment
* ex_050: nnを試す
* ex_051: minimam_nn(64-32-1) => CV: 0.755

# 2020/11/7
## experiment
* ex_048_3: 再度試す(速度改善ver)

## EDA
* ex_052:
  * PreviousAnswerにindexつける! (ex_048とコード自体はかわらない) => CV: 0.777(only model-0)
* 020_previous_xxx
  * part / content_id
    * ![image_24](image_24.png)
  * shift1_contentid: 正解したのにもう1回やってる人いる。単純に間違えただけでは? => ex_052: previous_answer3
    * ![image_25](image_25.png)
  * shift2_contentid: 間違えた! -> 教材見て -> もう1度復習　…これ、頑張ればquestionとlectureひも付けられそう
    * ![image_26](image_26.png)
  * shift3_contentid: shift2とおなじ(ちょっとタイムラグあるだけ)
    * ![image_27](image_27.png)
* 021_lecture-question
  * shift2_contentidで見つけたquestionとlectureをひも付けて、lectureを受けた人・受けてない人での正解率探す -> やっぱり受けた人のほうが正解率高い
    * ![image_28](image_28.png)

# 2020/11/8
## experiment
* ex_053: question_lecture_map
* ex_052_2: previous_answerのバグ直して再提出 -> LB: 0.780(down...)
* ex_054: catboost(feature: ex_052) depth tuning

# 2020/11/9
## experiment
* ex_052_2: indexリアルタイム更新にして再度提出 -> LB: 0.780...
* ex_055_0: PreviousAnswer、indexあり版でindex消すとCV0.770, indexなし版CV0.773
  * なぜ？データ確認する -> user_idのみでgroupbyしてたので、単純に「前の問題に正解してるか」になってた
* ex_055_0_1: answer_idxもバグ確認
* ex_055: indexなしで1回subする(=exp048)
  * 原因がindexなのかanswerなのかで切り分けたい+スコア更新したい
* ex_056: ex_055+cat&lgbm
* ex_057: ex_056+index
  * lgbm CV: 0.779 -> LB: 0.786
  * cat CV: 0.780 -> LB: 0.783
  * lgbm+cat CV: 0.781(たぶん)

# 2020/11/10
## eda
* 022_previous_content
  * previous_content_type_id 1個前がlectureのときのほうが正解率高い -> 特徴に追加
    * ![image_29](image_29.png)
  * 更に見ていくと, content_type_id=[1, 0]の順番でのconunt_encoderで、数が多いほど正解率が高い。コレは使えそう
    * ![image_30](image_30.png)

# 2020/11/11
## experiment
* ex_059: depth tuning
* ex_060: ex_057+previous_lecture -> CV: 0.7770(変化なし?)
* ex_061: ex_060+initial_weight/initial_score -> CV: 0.777(ex_060とあんま変わらず)
* ex_062: ex_057+content_level_encoder -> CV: 0.7778(first model)
* ex_057とex_049でuser_rate系の値が違う
  * df.sum(), df.values.sum()、後者はnp.nanがあるとsumもnanになるよ！
  * ex_057_2.pyでfeature_factory_manager作り直し

# 2020/11/12
## experiment
* ex_063: ex_062 + groupby(user_id).mean(content_level)
  * TODO: groupby(content_id).mean(user_id)もしたいけど、ライブラリの制約でできねぇ…改造するか
* ex_064: ex_063のdf_train->fit vs all_predict
  * データの順番変えたらそりゃ結果かわるよね -> データの順番も完全一致させる？ -> いや、そこはあわせてたっぽい

# 2020/11/13
## experiment
* ex_066: ex_057 + bugfix -> CV: 0.7773(model1) LB: 0.786
* ex_067: target_enc_content_idをuser_idなどでdiffとりまくる(userにとって簡単かどうか?) -> CV: 0.7780(model1)
* ex_068: ex_067 + FirstPart, FirstContentId -> CV: 0.7782(model1)
* ex_069: ex_068 + First1~5Ans -> CV: 0.7780(model1) down...
* ex_070: ex_068 + user情報からcontentの難しさを測るという発想なら、逆もありでは?
  * target_enc_user_idをcontent情報系で -> 0.7782(model1)
* ex_072: ex_070 + map
* ex_072_0: map作り直し(threshold=300->50)

## EDA
* 023_first_X_answered_correctly -> 特徴に加えてみる
  * ![image_31](image_31.png)
* 024_first_XXX
  * TOEICの目標点によって変わるのでは?
    * content_id=7900が圧倒的に多いっぽい
      * ![image_32](image_32.png)
    * part: 5が多いっぽい。ただcontent_id=7900はpart=1だけどね。。
      * ![image_33](image_33.png)

# 2020/11/14
## experiment
* ex_073: countencoderを全部抜く -> CV: 0.7779(model0) LB: 0.786
* ex_074: PreviousAnswer, 500問以上前はnp.nan! (ex_073と内容一緒) model1 only CV: 0.7772(-0.0007)
* ex_075: ex_066を20Mrow-10model
* ex_076: ex_074 + session加える, shiftdiffを消す(※SessionEncoderはall_predictしか書いてない) CV: 0.7785
* ex_077: ex_077 + previous3answer CV: 0.7785 (+ 0.0006)
* ex_078: ex_074 + intention_count, target_enc_content_id_mean

## EDA
* 025_session: 学習セッション
  * sessionごとの正解率：最初が低い。。けどこれはすでに捉えてそう
    * ![image_34](image_34.png)
  * sessionごとに今何問目？の正解率
    * 0~30は低いが、30~で多くなってる。。まあこれも捉えてるんかなぁ…(一番最初のsession影響?)
    * あと0は極端に低いな
      * ![image_35](image_35.png)
      * ![image_36](image_36.png)
* 026_prev_n_answer_correctly:
  * あんまりuser_idとの相関もないし、結構効きそうだぞコレは!ただ、fitがリアルタイムじゃないからリークしてるかもなぁ
    * ![image_37](image_37.png)

## other
* GCP instance -> あきらめ…
  * https://qiita.com/inumaru9n/items/e8250af967431c46ad45

# 2020/11/15
## experiment
* ex_078: Counter(intention), TargetEncAggわちゃわち -> 0.7782
* ex_079: 特徴カット(ex_074 base) -> 0.77719
* ex_080: ql_map -> 0.77910
* ex_081: ex_076-ex_080をあわせた -> 0.77938
* ex_082: ex_081のTargetEncをcontent_idにしぼりこみ -> 0.77938(いっしょ)
* ex_083: Session抜く -> 0.77887
* ex_084: Session抜いてとりあえず10model -> CV: 0.7786(model1) LB: 0.782 (-0.004)
* ex_085: ex_084の20Mrow version

## EDA
* 027_lecture_typeof
  * intentionはちょっと効くかな?って感じ
  * ![image_38](image_38.png)

## other
* performance tuning.
  * fit
      * ContentLevelEncoder 90ms -> 50ms
      * CategoryLevelEncoder 70+80ms -> 10+10ms

# 2020/11/16
## experiment
* ex_086: te_content - (te_user系), te_user - (te_content系) -> CV: 0.7779

# 2020/11/17
## experiment
* ex_088: bug check
  * te_xxx: df.values.mean とかにしてた. numpyはnullが１個でもあるとだめだよ
  * previousNanswer: なんかバグってる -> partial_predictが更新ありの場合、is_partial_fit=Trueだと２回更新されちゃう
* ex_084をもう一度(CVのみ) -> CV: 0.7785(model1, -0.0001)
* ex_089: QLtable2
  * 精度変わらず
    * QLtableは0-1の引き算, QLtable2は生値？ -> QLtable2を引き算に変更
    * よくみたら単純にバグってただけ…
    * QLtable diff -> model1 CV: 0.7798(100epoch)-0.7805(200epoch)
    * QLtable normal -> model1 CV: 0.7832(100epoch)-0.7841(200epoch)
* ex_084: もう一度, fitだけしてサブする
* ex_090: ex_089を20Mrow(GCP)

## other
* 処理の並列化
  * https://qiita.com/uo_ta/items/8d265a0f03300ebc2635
</div>
