客服对话记忆
==============

记住用户的历史订单、偏好和投诉记录，提供个性化服务。

完整代码（可直接运行）
-----------------------

可运行示例：``examples/customer_support/run_customer_support.py``

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem.from_yaml_config("config.yaml")

   conversations = [
       ("cs_1", "我想退昨天买的蓝色运动鞋，尺码不合适", "2024-03-10T14:00:00"),
       ("cs_2", "好的，已为您提交退货申请，预计3-5个工作日退款", "2024-03-10T14:01:00"),
       ("cs_3", "下次我想买42码的同一款，有货吗？", "2024-03-10T14:02:00"),
   ]
   for uid, text, ts in conversations:
       system.add(MemoryUnit(
           uid=Uid(uid),
           raw_data={"text_content": text},
           metadata={"timestamp": ts},
       ))

   system.build_high_level(mode="auto")

   # 全局检索
   hits = system.holistic_retrieve("这个客户之前买了什么鞋？", top_k=3)
   # 知识视角检索
   knowledge = system.retrieve_by_view("客户喜欢什么？", view="knowledge", top_k=5)
   # 事件因果检索
   events = system.retrieve_by_view("退货的原因是什么？", view="event_causal", top_k=5)

   system.save("./cs_memory")

预期输出对照
-----------

**holistic_retrieve("这个客户之前买了什么鞋？")**：

.. code-block::

   [0.941] 我想退昨天买的蓝色运动鞋，尺码不合适

**retrieve_by_view("客户喜欢什么？", view="knowledge")**：

.. code-block::

   [0.887] 知识实体: 蓝色运动鞋 - 用户偏好品牌/款式，尺码偏好42码

**retrieve_by_view("退货的原因是什么？", view="event_causal")**：

.. code-block::

   [0.912] 事件因果: 尺码不合适 → 退货申请

Mandol 在客服场景中的价值
--------------------------

- **实体自动提取**：无需手动标注，自动从对话中提取商品名、尺码
- **事件因果链**：自动构建「尺码不合适→退货」的因果关系
- **跨会话记忆**：同一用户多次咨询时能检索到历史对话
