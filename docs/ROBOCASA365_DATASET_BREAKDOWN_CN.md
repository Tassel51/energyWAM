# RoboCasa365 本机数据与任务划分明细

统计日期：2026-11-01。单位均为完整示范轨迹（episode）。中文为任务含义概述；具体物品和指令以每条轨迹为准。

来源：RoboCasa365 target/human；本机通过 HF 镜像 ryanhoangt/robocasa365_datasets_split_target_source_human 下载。原始目录：/datasets/robocasa365/v1.0/target/。

划分：在每个任务内部，原始 episode_index % 20 为 0 时进入测试集，为 1 时进入验证集，其余进入训练集。这是本项目的轨迹级划分，不是官方另行提供的训练/验证/测试包，也不保证三个集合的厨房场景互斥。

官方介绍：https://robocasa.ai/docs/build/html/datasets/datasets_overview.html

## 基础操作：18 个任务

|任务含义|任务标识|总数|训练|验证|测试|
|---|---|---:|---:|---:|---:|
|盖好搅拌机盖|CloseBlenderLid|502|450|26|26|
|关闭冰箱门|CloseFridge|513|461|26|26|
|关闭台式烤箱门|CloseToasterOvenDoor|505|453|26|26|
|把杯子放到咖啡机出水口下|CoffeeSetupMug|502|450|26|26|
|在厨房移动到指定位置|NavigateKitchen|500|450|25|25|
|打开橱柜门|OpenCabinet|500|450|25|25|
|打开抽屉|OpenDrawer|514|462|26|26|
|抬起厨师机机头|OpenStandMixerHead|502|450|26|26|
|把台面物品放进橱柜|PickPlaceCounterToCabinet|502|450|26|26|
|把台面物品放到炉灶上的锅具中|PickPlaceCounterToStove|501|450|25|26|
|把抽屉物品拿到台面|PickPlaceDrawerToCounter|501|450|25|26|
|把水槽物品拿到台面|PickPlaceSinkToCounter|501|450|25|26|
|把烤好的物品取到台面餐盘|PickPlaceToasterToCounter|512|460|26|26|
|推拉洗碗机碗架|SlideDishwasherRack|502|450|26|26|
|关闭炉灶火头|TurnOffStove|500|450|25|25|
|打开电热水壶开关|TurnOnElectricKettle|520|468|26|26|
|启动微波炉|TurnOnMicrowave|543|487|28|28|
|打开水龙头|TurnOnSinkFaucet|506|454|26|26|
|合计||9126|8195|464|467|

## 组合任务：32 个任务

|任务含义|任务标识|总数|训练|验证|测试|
|---|---|---:|---:|---:|---:|
|整理面包篮并送到餐台|ArrangeBreadBasket|514|462|26|26|
|把水壶和杯子摆到托盘|ArrangeTea|500|450|25|25|
|挑选指定面包并搭配果酱|BreadSelection|503|451|26|26|
|按类别归放调味品|CategorizeCondiments|513|461|26|26|
|选择适合的切削工具|CuttingToolSelection|501|450|25|26|
|从抽屉取吸管并放入杯中|DeliverStraw|504|452|26|26|
|取水果装饰煎饼|GarnishPancake|503|451|26|26|
|收集餐具并分类放入橱柜|GatherTableware|509|457|26|26|
|启动烤面包机并取出烤好面包|GetToastedBread|506|454|26|26|
|用台式烤箱加热烤串和面包|HeatKebabSandwich|503|451|26|26|
|把水壶放上炉灶并开火|KettleBoiling|501|450|25|26|
|把杯碗装进洗碗机并关门|LoadDishwasher|501|450|25|26|
|往柠檬水中加入柠檬和冰块|MakeIceLemonade|512|460|26|26|
|分装两份相同午餐|PackIdenticalLunches|501|450|25|26|
|把锅中食物倒入盘中并归还锅|PanTransfer|510|458|26|26|
|给各个盘子分配热狗面包和香肠|PortionHotDogs|504|452|26|26|
|把锅和海绵放进水槽并放水浸泡|PreSoakPan|501|450|25|26|
|取杯并启动咖啡机制作咖啡|PrepareCoffee|514|462|26|26|
|按塑料和玻璃分类瓶子|RecycleBottlesByType|520|468|26|26|
|开水并移动出水口冲洗水槽|RinseSinkBasin|509|457|26|26|
|用海绵擦洗砧板|ScrubCuttingBoard|504|452|26|26|
|准备锅和肉并开火煎肉|SearingMeat|501|450|25|26|
|把肉类和蔬菜容器分层放入冷冻室|SeparateFreezerRack|501|450|25|26|
|取刀具和食材布置切菜区|SetUpCuttingStation|504|452|26|26|
|在橱柜中叠放碗|StackBowlsCabinet|515|463|26|26|
|把食物装碗并放入微波炉加热|SteamInMicrowave|511|459|26|26|
|把蔬菜放入锅并用锅铲搅拌|StirVegetables|501|450|25|26|
|把剩余食物装碗后放入冰箱|StoreLeftoversInBowl|503|451|26|26|
|用微波炉重新加热华夫饼|WaffleReheat|500|450|25|25|
|把水果放入滤篮并冲洗|WashFruitColander|507|455|26|26|
|在水槽中冲洗生菜|WashLettuce|501|450|25|26|
|把食材放到电子秤上称重|WeighIngredients|504|452|26|26|
|合计||16181|14530|821|830|

总计：50 个任务，25,307 条轨迹；训练 22,725，验证 1,285，测试 1,297；共 14,957,899 帧。

官方概述使用约 500 条/任务的规模描述；上述是本机文件逐条核对得到的准确数量。数量差异的具体采集原因没有从元数据中确认。
