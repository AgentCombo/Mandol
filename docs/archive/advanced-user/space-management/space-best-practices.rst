空间管理最佳实践
==================

多用户空间隔离
--------------

.. code-block:: python

   users = ["user_001", "user_002", "user_003"]

   for uid in users:
       system.semantic_map.create_space(f"用户-{uid}")
       user_units = load_user_history(uid)
       system.add_many(user_units, space_names=[f"用户-{uid}"])
       system.build_high_level(mode="auto")

   # 只查某个用户的记忆
   hits = system.retrieve_in_space(
       "最近的订单", space_name="用户-001"
   )

业务空间分层
------------

.. code-block:: python

   system.semantic_map.create_space("业务")
   system.semantic_map.ensure_child_space("业务", "订单")
   system.semantic_map.ensure_child_space("业务", "售后")
   system.semantic_map.ensure_child_space("业务", "咨询")

   # 客服对话按业务类型归类
   system.add(order_unit, space_names=["业务/订单"])
   system.add(after_sale_unit, space_names=["业务/售后"])

   # 查询某类业务的全部记忆
   hits = system.retrieve_in_space(
       "用户反馈", space_name="业务/售后"
   )

临空间与持久空间
-----------------

.. code-block:: python

   # 临时分析空间（用完即删）
   system.semantic_map.create_space("分析-临时")
   system.add_many(analysis_units, space_names=["分析-临时"])
   hits = system.retrieve_in_space("模式", space_name="分析-临时")
   # ... 分析完成后 ...
   system.semantic_map.delete_space("分析-临时", cascade=True)

空间过多时的性能提示
---------------------

- 100+ 空间：基本无影响
- 1000+ 空间：建议定期清理不再使用的空空间
- ``list_spaces()`` 返回完整列表，大量空间时考虑用 ``get_space()`` 按需获取
