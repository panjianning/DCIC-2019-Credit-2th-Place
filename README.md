# DCIC-2019-Credit-2th-Place
2019数字中国创新大赛 消费者人群画像 亚军 

NLP队不完整代码（只包含我这部分）。

#### util.py： 一些工具函数
封装了lightgbm, catboost等，方便K折，且自定义了一些损失函数。

#### gotcha_lgb.ipynb： lightgbm模型

```
manual_feature = [
 '当月费用-前五个月消费平均费用',
 '前五个月消费总费用',
 'count_缴费',
 'count_当月费用',
 'count_费用差',
 'count_平均费用',
 'count_当月费用_平均费用'
 '是否998折']
```
* lgb1: 原始特征+manual_feature， loss为 `0.5*mae+0.5*fair(fair_c=25)`
* lgb2: 原始特征+manual_feature(年龄0变换为nan), loss为`0.5*huber(delta=2)+0.5*fair(fair_c=23)`
* lgb3: 原始特征+manual_feature(app次数特征做了round_log1p变换))， loss为`0.5*mae+0.5*fair(fair_c=25)`
* lgb4: 原始特征+manual_feature(年龄0变换为nan， app次数特征做了round_log1p变换), loss为`0.5*huber(delta=2)+0.5*fair(fair_c=23)`
* lgb5: 原始特征+['当月费用-前五个月消费平均费用']. loss为`0.5*huber(delta=2)+0.5*fair(fair_c=23)`
* lgb6: 原始特征+manual_feature， loss为 `0.5*mae+0.5*fair(fair_c=25)`， target做了`np.power(1.005, x)`变换(idea from @Neil)


#### gotcha_gbdt.ipynb： sklearn gbdt模型
* gbdt1:  原始特征+manual_feature, loss为huber

#### gotcha_ctb.ipynb： sklearn ctb模型
* ctb1:  原始特征+manual_feature, loss为mae, target做了`np.power(1.005, x)`变换(idea from @Neil)

#### stacking.ipynb
将所有模型的结果用huber_regressor做stacking

#### 关于自定义loss
可参考
* [Effect of MAE](https://www.kaggle.com/c/allstate-claims-severity/discussion/24520)
* [Xgboost-How to use “mae” as objective function?](https://stackoverflow.com/questions/45006341/xgboost-how-to-use-mae-as-objective-function)
