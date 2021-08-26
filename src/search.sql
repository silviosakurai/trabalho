SELECT COUNT(DISTINCT T2.seller_id),
    MAX(T1.order_approved_at),
    MIN(T1.order_approved_at)

FROM fr_orders AS T1

LEFT JOIN fr_orders_items as T2
ON T1.order_id = T2.order_id

WHERE T1.order_approved_at BETWEEN '2017-06-01' and '2018-06-01'