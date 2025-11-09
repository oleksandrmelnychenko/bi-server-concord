# V3.2 Quality Improvements Implementation Plan

**Date:** November 9, 2025
**Version:** V3.2 (Enhanced Discovery Quality)
**Status:** 🚧 In Progress

---

## 🎯 Quality Improvements Summary

### 1. ✅ **Product Groups Discovered**
- Found `ProductProductGroup` table with 100% coverage (369,742 products)
- Product groups available: BOSCH, FEBI, AUGER, HELLA, Diesel Technic, etc.
- Can implement diversity filtering

---

## 📋 V3.2 Implementation Checklist

### **Priority 1: Weighted Similarity Algorithm** ⏳

**Current (V3.1):**
```python
# Simple Jaccard similarity
similarity = len(A ∩ B) / len(A ∪ B)
```

**New (V3.2):**
```python
# Weighted collaborative filtering
similarity = (
    0.5 * jaccard_similarity(products_A, products_B) +
    0.3 * recency_similarity(last_order_A, last_order_B) +
    0.2 * frequency_similarity(order_rate_A, order_rate_B)
)
```

**Changes needed:**
1. Add `get_customer_order_stats()` method:
   ```python
   def get_customer_order_stats(self, customer_id, as_of_date):
       """Get customer's order recency and frequency"""
       # Return: (days_since_last_order, orders_per_month)
   ```

2. Add `calculate_recency_similarity()`:
   ```python
   def calculate_recency_similarity(recency_A, recency_B):
       """
       Compare order recency between two customers
       - Similar if both ordered recently
       - Returns 0-1 score
       """
   ```

3. Add `calculate_frequency_similarity()`:
   ```python
   def calculate_frequency_similarity(freq_A, freq_B):
       """
       Compare order frequency between two customers
       - Similar if both order at similar rates
       - Returns 0-1 score
       """
   ```

4. Modify `find_similar_customers()` to use weighted score

---

###  **Priority 2: Trending Products Boost** ⏳

**Implementation:**
1. Add `get_trending_products()` method:
   ```python
   def get_trending_products(self, days=7, growth_threshold=0.5):
       """
       Find products with 50%+ order growth in last 7 days

       Returns: Set of trending product IDs
       """
       query = """
       SELECT
           ProductID,
           COUNT(CASE WHEN o.Created >= DATEADD(day, -7, GETDATE()) THEN 1 END) as recent_orders,
           COUNT(CASE WHEN o.Created >= DATEADD(day, -30, GETDATE())
                      AND o.Created < DATEADD(day, -7, GETDATE()) THEN 1 END) as baseline_orders
       FROM OrderItem oi
       JOIN [Order] o ON oi.OrderID = o.ID
       WHERE o.Created >= DATEADD(day, -30, GETDATE())
       GROUP BY ProductID
       HAVING COUNT(CASE WHEN o.Created >= DATEADD(day, -7, GETDATE()) THEN 1 END) > 5
           AND COUNT(CASE WHEN o.Created >= DATEADD(day, -30, GETDATE())
                          AND o.Created < DATEADD(day, -7, GETDATE()) THEN 1 END) > 0
           AND CAST(COUNT(CASE WHEN o.Created >= DATEADD(day, -7, GETDATE()) THEN 1 END) AS FLOAT) /
               NULLIF(COUNT(CASE WHEN o.Created >= DATEADD(day, -30, GETDATE())
                                 AND o.Created < DATEADD(day, -7, GETDATE()) THEN 1 END) / 3.0, 0) > (1 + {growth_threshold})
       """
   ```

2. Cache trending products (compute once per day)

3. Apply boost in `get_recommendations()`:
   ```python
   if product_id in trending_products:
       score *= 1.2  # 20% boost for trending products
   ```

---

### **Priority 3: Remove 'Skip Heavy Users' Optimization** ⏳

**Current (V3.1):**
```python
# Skip discovery for HEAVY users (performance optimization)
if include_discovery and segment != "HEAVY":
    collaborative_scores = self.get_collaborative_score(customer_id, as_of_date)
```

**New (V3.2):**
```python
# ALL customers get discovery (including HEAVY)
if include_discovery:
    collaborative_scores = self.get_collaborative_score(customer_id, as_of_date)
```

**Impact:** Heavy users will now get 5-10 new product recommendations

---

### **Priority 4: Enforce Strict Old/New Mix** ⏳

**Current (V3.1):**
```python
# Blends repurchase and discovery based on segment weights
# Result: Variable mix (Heavy: 25/0, Light: 10/15)
```

**New (V3.2):**
```python
def get_recommendations(self, customer_id, as_of_date, top_n=25,
                       old_products=20, new_products=5):
    """
    Enforce strict old/new mix:
    - old_products: 15-20 repurchase products (default: 20)
    - new_products: 5-10 discovery products (default: 5)
    """

    # Get repurchase recommendations
    repurchase_recs = self.get_repurchase_products(
        customer_id, as_of_date, top_n=old_products
    )

    # Get discovery recommendations (exclude already owned)
    owned_products = {r['product_id'] for r in repurchase_recs}
    discovery_recs = self.get_discovery_products(
        customer_id, as_of_date,
        top_n=new_products,
        exclude=owned_products
    )

    # Combine: 20 old + 5 new = 25 total
    return repurchase_recs + discovery_recs
```

---

### **Priority 5: Product Group Diversity Filter** ⏳

**Implementation:**
1. Add `get_product_groups()` method:
   ```python
   def get_product_groups(self, product_ids):
       """
       Get product group mapping for given products

       Returns: {product_id: group_id, ...}
       """
       query = """
       SELECT ProductID, ProductGroupID
       FROM ProductProductGroup
       WHERE ProductID IN ({})
       AND Deleted = 0
       """.format(','.join(map(str, product_ids)))
   ```

2. Add `apply_diversity_filter()`:
   ```python
   def apply_diversity_filter(self, recommendations, max_per_group=3):
       """
       Ensure diversity across product groups

       Rules:
       - Max 3 products per group
       - Ensure at least 3 different groups
       """
       # Get product groups
       product_ids = [r['product_id'] for r in recommendations]
       groups = self.get_product_groups(product_ids)

       # Count products per group
       group_counts = defaultdict(int)
       filtered = []

       for rec in recommendations:
           group_id = groups.get(rec['product_id'])
           if group_counts[group_id] < max_per_group:
               filtered.append(rec)
               group_counts[group_id] += 1

       return filtered
   ```

3. Apply in `get_recommendations()` before returning

---

## 📊 Expected Performance Impact

| Metric | V3.1 | V3.2 (Estimated) | Change |
|--------|------|------------------|--------|
| **Heavy user latency** | 248ms (no discovery) | ~2,000ms (with discovery) | ⚠️ 8x slower |
| **Light user latency** | 1,589ms | ~1,800ms (weighted sim) | ~13% slower |
| **Discovery quality** | Good | **Better** (weighted sim) | ✅ Improved |
| **Discovery for Heavy** | 0 products | 5 products | ✅ Added |
| **Product diversity** | None | Max 3 per group | ✅ Added |
| **Trending awareness** | None | 20% boost | ✅ Added |

---

## 🧪 Testing Plan

### Test 1: Weighted Similarity
```python
# Compare V3.1 vs V3.2 similar customers for same customer
customer_id = 410176
v31_similar = v31.find_similar_customers(customer_id, '2024-07-01')
v32_similar = v32.find_similar_customers(customer_id, '2024-07-01')

# Expect: Different similar customers (weighted by recency/frequency)
```

### Test 2: Old/New Mix
```python
# Test that ALL customers get exactly 20 old + 5 new
for customer in [410169, 410175, 410176, 410180]:
    recs = v32.get_recommendations(customer, '2024-07-01', top_n=25)
    old_count = sum(1 for r in recs if r['source'] == 'repurchase')
    new_count = sum(1 for r in recs if r['source'] == 'discovery')

    assert old_count == 20, f"Expected 20 old, got {old_count}"
    assert new_count == 5, f"Expected 5 new, got {new_count}"
```

### Test 3: Heavy Users Get Discovery
```python
# Test that Heavy users now get discovery products
customer_id = 410169  # HEAVY user
recs = v32.get_recommendations(customer_id, '2024-07-01', top_n=25)
discovery_count = sum(1 for r in recs if r['source'] == 'discovery')

assert discovery_count == 5, f"Heavy user should get 5 discovery, got {discovery_count}"
```

### Test 4: Product Diversity
```python
# Test that no more than 3 products from same group
recs = v32.get_recommendations(410169, '2024-07-01', top_n=25)
product_ids = [r['product_id'] for r in recs]
groups = v32.get_product_groups(product_ids)

from collections import Counter
group_counts = Counter(groups.values())
max_count = max(group_counts.values())

assert max_count <= 3, f"Max products per group should be 3, got {max_count}"
```

### Test 5: Trending Boost
```python
# Get trending products
trending = v32.get_trending_products(days=7, growth_threshold=0.5)
print(f"Found {len(trending)} trending products")

# Check if trending products appear in recommendations
recs = v32.get_recommendations(410169, '2024-07-01', top_n=25)
trending_in_recs = [r for r in recs if r['product_id'] in trending]
print(f"Trending products in recommendations: {len(trending_in_recs)}")
```

---

## ⚠️ Known Issues & Considerations

### Issue 1: Heavy User Performance Degradation
**Problem:** Heavy users will be 8x slower (248ms → 2,000ms)

**Mitigation Options:**
1. **Accept it:** 2s is still acceptable for weekly pre-computation
2. **Limit similar customers:** Reduce from 100 to 50 for Heavy users
3. **Pre-compute similar customers:** Nightly job to cache similarities

**Recommendation:** Accept for now (weekly job), optimize later if needed

### Issue 2: Product Diversity May Reduce Precision
**Problem:** Forcing diversity might exclude highly relevant products

**Mitigation:**
- Make `max_per_group` configurable (default: 3)
- Apply diversity only to discovery products, not repurchase
- Monitor precision impact in A/B test

### Issue 3: Trending Products Cache Staleness
**Problem:** Trending products computed daily, might miss hourly trends

**Mitigation:**
- Acceptable for weekly recommendations
- For real-time API, compute trending hourly
- Cache with 1-hour TTL

---

## 📝 Implementation Status

- [x] Database exploration (product groups found)
- [ ] Add weighted similarity algorithm
- [ ] Add trending products boost
- [ ] Remove 'skip Heavy users' optimization
- [ ] Enforce strict old/new mix
- [ ] Add product diversity filter
- [ ] Test on sample customers
- [ ] Update documentation
- [ ] Deploy to production

---

## 🚀 Next Steps

1. **Implement V3.2:** Make all code changes in `improved_hybrid_recommender_v32.py`
2. **Test thoroughly:** Run all 5 test cases
3. **Compare V3.1 vs V3.2:** Quality and performance
4. **Update API:** Switch from V3.1 to V3.2
5. **Update worker:** Use V3.2 for weekly recommendations
6. **Monitor production:** Track precision, latency, discovery rate

---

**Status:** 🚧 Ready to implement
**Next:** Start coding V3.2 with all quality improvements
