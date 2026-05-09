场景一：客服对话记忆系统
========================

.. note::

   本文档已迁移至 :doc:`/basic-user/scenarios/customer-support`。本页面将在后续版本中移除，请更新你的书签。

本场景演示如何使用 Mandol 构建电商客服对话记忆系统，记住用户的历史订单、偏好、投诉记录，提供个性化服务。

场景说明
--------

在电商客服场景中，同一用户可能在不同时间发起多次咨询。Mandol 能够：

- 自动提取用户提到的商品、订单、偏好等实体
- 记录退货、投诉等事件及其因果关系
- 在后续咨询时精准检索用户历史信息

完整代码示例
------------

可运行示例：``examples/customer_support/run_customer_support.py``

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem.from_yaml_config("config.yaml")

   conversations = [
       MemoryUnit(
           uid=Uid("cs_001"),
           raw_data={"text_content": "我想退昨天买的蓝色运动鞋，尺码不合适"},
           metadata={"timestamp": "2024-03-10T14:00:00", "speaker": "customer"},
       ),
       MemoryUnit(
           uid=Uid("cs_002"),
           raw_data={"text_content": "好的，已为您提交退货申请，预计3-5个工作日退款"},
           metadata={"timestamp": "2024-03-10T14:01:00", "speaker": "agent"},
       ),
       MemoryUnit(
           uid=Uid("cs_003"),
           raw_data={"text_content": "下次我想买42码的同一款，有货吗？"},
           metadata={"timestamp": "2024-03-10T14:02:00", "speaker": "customer"},
       ),
   ]
   for unit in conversations:
       system.add(unit)

   report = system.build_high_level(mode="auto")

   hits = system.holistic_retrieve("这个客户之前买了什么鞋？", top_k=3)

   preferences = system.retrieve_by_view("客户喜欢什么？", view="knowledge", top_k=5)

   events = system.retrieve_by_view("退货的原因是什么？", view="event_causal", top_k=5)

   system.save("./cs_memory")

检索结果说明
^^^^^^^^^^^^

**holistic_retrieve("这个客户之前买了什么鞋？")**

全记忆检索会从 BASE / ENTITY / EVENT / SUMMARY 四组中召回相关结果。预期返回关于蓝色运动鞋购买的对话记录，命中 ``cs_001`` 等单元。

**retrieve_by_view("客户喜欢什么？", view="knowledge")**

知识视角检索会从知识实体空间中查找用户偏好。预期返回"蓝色运动鞋"相关的知识实体摘要，包含用户的尺码偏好（42码）。

**retrieve_by_view("退货的原因是什么？", view="event_causal")**

事件因果视角检索会从事件空间中查找因果关系链。预期返回"尺码不合适 → 退货"的因果链事件。

为什么选择 Mandol？
-------------------

1. **实体自动提取**：无需手动标注，LLM 自动从对话中提取商品名、尺码、订单号等实体
2. **事件因果链**：自动构建"尺码不合适→退货"的因果关系，支持因果推理类查询
3. **跨会话记忆**：同一用户多次咨询时，系统能检索到历史对话中的关键信息
4. **多视角检索**：同一份记忆数据支持从知识、事件、情感等不同视角检索，满足不同业务需求
