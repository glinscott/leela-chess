/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include "utils/mutex.h"

namespace lczero {

// Generic LRU cache. Thread-safe.
template <class K, class V>
class LruCache {
  static const double constexpr kLoadFactor = 1.33;

 public:
  LruCache(int capacity = 128)
      : capacity_(capacity), hash_(capacity * kLoadFactor + 1) {
    std::memset(&hash_[0], 0, sizeof(hash_[0]) * hash_.size());
  }

  ~LruCache() {
    ShrinkToCapacity(0);
    assert(size_ == 0);
    assert(allocated_ == 0);
  }

  // Inserts the element under key @key with value @val.
  // If the element is pinned, old value is still kept (until fully unpinned),
  // but new lookups will return updated value.
  // If @pinned, pins inserted element, Unpin has to be called to unpin.
  // In any case, puts element to front of the queue (makes it last to evict).
  V* Insert(K key, std::unique_ptr<V> val, bool pinned = false) {
    Mutex::Lock lock(mutex_);

    auto hash = hasher_(key) % hash_.size();
    auto& hash_head = hash_[hash];
    for (Item* iter = hash_head; iter; iter = iter->next_in_hash) {
      if (key == iter->key) {
        EvictItem(iter);
        break;
      }
    }

    ShrinkToCapacity(capacity_ - 1);
    ++size_;
    ++allocated_;
    Item* new_item = new Item(key, std::move(val), pinned ? 1 : 0);
    new_item->next_in_hash = hash_head;
    hash_head = new_item;
    InsertIntoLru(new_item);
    return new_item->value.get();
  }

  // Checks whether a key exists. Doesn't lock. Of course the next moment the
  // key may be evicted.
  bool ContainsKey(K key) {
    Mutex::Lock lock(mutex_);
    auto hash = hasher_(key) % hash_.size();
    for (Item* iter = hash_[hash]; iter; iter = iter->next_in_hash) {
      if (key == iter->key) return true;
    }
    return false;
  }

  // Looks up and pins the element by key. Returns nullptr if not found.
  // If found, brings the element to the head of the queue (makes it last to
  // evict).
  V* Lookup(K key) {
    Mutex::Lock lock(mutex_);

    auto hash = hasher_(key) % hash_.size();
    for (Item* iter = hash_[hash]; iter; iter = iter->next_in_hash) {
      if (key == iter->key) {
        // BringToFront(iter);
        ++iter->pins;
        return iter->value.get();
      }
    }
    return nullptr;
  }

  // Unpins the element given key and value.
  void Unpin(K key, V* value) {
    Mutex::Lock lock(mutex_);

    // Checking evicted list first.
    Item** cur = &evicted_head_;
    for (Item* el = evicted_head_; el; el = el->next_in_hash) {
      if (key == el->key && value == el->value.get()) {
        if (--el->pins == 0) {
          *cur = el->next_in_hash;
          --allocated_;
          delete el;
        }
        return;
      }
      cur = &el->next_in_hash;
    }

    // Now lookup in actve list.
    auto hash = hasher_(key) % hash_.size();
    for (Item* iter = hash_[hash]; iter; iter = iter->next_in_hash) {
      if (key == iter->key && value == iter->value.get()) {
        assert(iter->pins > 0);
        --iter->pins;
        return;
      }
    }
    assert(false);
  }

  // Sets the capacity of the cache. If new capacity is less than current size
  // of the cache, oldest entries are evicted. In any case the hashtable is
  // rehashed.
  void SetCapacity(int capacity) {
    Mutex::Lock lock(mutex_);

    if (capacity_ == capacity) return;
    ShrinkToCapacity(capacity);
    capacity_ = capacity;

    std::vector<Item*> new_hash(capacity * kLoadFactor + 1);
    std::memset(&new_hash[0], 0, sizeof(new_hash[0]) * new_hash.size());

    if (size_ != 0) {
      for (Item* head : hash_) {
        for (Item* iter = head; head; head = head->next_in_hash) {
          auto& new_hash_head = new_hash[hasher_(iter->key) % new_hash.size()];
          iter->next_in_hash = new_hash_head;
          new_hash_head = iter;
        }
      }
    }
    hash_.swap(new_hash);
  }

  int GetSize() const {
    Mutex::Lock lock(mutex_);
    return size_;
  }
  int GetCapacity() const {
    Mutex::Lock lock(mutex_);
    return capacity_;
  }

 private:
  struct Item {
    Item(K key, std::unique_ptr<V> value, int pins)
        : key(key), value(std::move(value)), pins(pins) {}
    K key;
    std::unique_ptr<V> value;
    int pins = 0;
    Item* next_in_hash = nullptr;
    Item* prev_in_queue = nullptr;
    Item* next_in_queue = nullptr;
  };

  void EvictItem(Item* iter) REQUIRES(mutex_) {
    --size_;

    // Remove from LRU list.
    if (lru_head_ == iter) {
      lru_head_ = iter->next_in_queue;
    } else {
      iter->prev_in_queue->next_in_queue = iter->next_in_queue;
    }
    if (lru_tail_ == iter) {
      lru_tail_ = iter->prev_in_queue;
    } else {
      iter->next_in_queue->prev_in_queue = iter->prev_in_queue;
    }

    // Destroy or move into evicted list dependending on whether it's pinned.
    Item** cur = &hash_[hasher_(iter->key) % hash_.size()];
    for (Item* el = *cur; el; el = el->next_in_hash) {
      if (el == iter) {
        *cur = el->next_in_hash;
        if (el->pins == 0) {
          --allocated_;
          delete el;
        } else {
          el->next_in_hash = evicted_head_;
          evicted_head_ = el;
        }
        return;
      }
      cur = &el->next_in_hash;
    }

    assert(false);
  }

  void ShrinkToCapacity(int capacity) REQUIRES(mutex_) {
    if (capacity < 0) capacity = 0;
    while (lru_tail_ && size_ > capacity) {
      EvictItem(lru_tail_);
    }
  }

  void BringToFront(Item* iter) REQUIRES(mutex_) {
    if (lru_head_ == iter) {
      return;
    } else {
      iter->prev_in_queue->next_in_queue = iter->next_in_queue;
    }
    if (lru_tail_ == iter) {
      lru_tail_ = iter->prev_in_queue;
    } else {
      iter->next_in_queue->prev_in_queue = iter->prev_in_queue;
    }

    InsertIntoLru(iter);
  }

  void InsertIntoLru(Item* iter) REQUIRES(mutex_) {
    iter->next_in_queue = lru_head_;
    iter->prev_in_queue = nullptr;

    if (lru_head_) {
      lru_head_->prev_in_queue = iter;
    }
    lru_head_ = iter;
    if (lru_tail_ == nullptr) {
      lru_tail_ = iter;
    }
  }

  // Fresh in front, stale on back.
  int capacity_ GUARDED_BY(mutex_);
  int size_ GUARDED_BY(mutex_) = 0;
  int allocated_ GUARDED_BY(mutex_) = 0;
  Item* lru_head_ GUARDED_BY(mutex_) = nullptr;  // Newest elements.
  Item* lru_tail_ GUARDED_BY(mutex_) = nullptr;  // Oldest elements.
  Item* evicted_head_ GUARDED_BY(mutex_) =
      nullptr;  // Evicted but pinned elements.
  std::vector<Item*> hash_ GUARDED_BY(mutex_);
  std::hash<K> hasher_ GUARDED_BY(mutex_);

  mutable Mutex mutex_;
};

// Convenience class for pinning cache items.
template <class K, class V>
class LruCacheLock {
 public:
  // Looks up the value in @cache by @key and pins it if found.
  LruCacheLock(LruCache<K, V>* cache, K key)
      : cache_(cache), key_(key), value_(cache->Lookup(key_)) {}

  // Unpins the cache entry (if holds).
  ~LruCacheLock() {
    if (value_) cache_->Unpin(key_, value_);
  }

  // Returns whether lock holds any value.
  operator bool() const { return value_; }

  // Gets the value.
  V* operator->() const { return value_; }
  V* operator*() const { return value_; }

  LruCacheLock() {}
  LruCacheLock(LruCacheLock&& other)
      : cache_(other.cache_), key_(other.key_), value_(other.value_) {
    other.value_ = nullptr;
  }
  void operator=(LruCacheLock&& other) {
    cache_ = other.cache_;
    key_ = other.key_;
    value_ = other.value_;
    other.value_ = nullptr;
  }

 private:
  LruCache<K, V>* cache_ = nullptr;
  K key_;
  V* value_ = nullptr;
};

}  // namespace lczero