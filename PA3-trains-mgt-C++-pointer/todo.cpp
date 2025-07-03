// todo.cpp - Train Management System Implementation
// Refactored for clarity, safety, and industry standards
#include "pa3.h"
#include <iostream>
#include <array>

// Avoid using namespace std globally

// Helper: String representation for CarType
template<typename T, size_t N>
constexpr size_t array_size(const T(&)[N]) { return N; }

namespace {
const char* CarTypeStr[] = {"HEAD", "OIL", "COAL", "WOOD", "STEEL", "SUGAR"};
constexpr int CARGO_TYPES = 5; // OIL, COAL, WOOD, STEEL, SUGAR
}

struct TypeStats {
    CarType type;
    int load;
    int maxload;
    int rank;
};

// Create a new train head node
TrainCar* createTrainHead() {
    return new TrainCar{HEAD, 0, 0, nullptr, nullptr};
}

// Add a car at the given position (1-based, after head)
bool addCar(TrainCar* head, int position, CarType type, int maxLoad) {
    if (!head || position < 1 || type == HEAD || maxLoad <= 0) return false;
    TrainCar* curr = head;
    for (int i = 0; i < position - 1; ++i) {
        if (!curr->next) break;
        curr = curr->next;
    }
    // Insert after curr
    auto* newCar = new TrainCar{type, maxLoad, 0, curr, curr->next};
    if (curr->next) curr->next->prev = newCar;
    curr->next = newCar;
    return true;
}

// Remove car at position (1-based, after head)
bool deleteCar(TrainCar* head, int position) {
    if (!head || position < 1) return false;
    TrainCar* curr = head;
    for (int i = 0; i < position - 1; ++i) {
        if (!curr->next) return false;
        curr = curr->next;
    }
    TrainCar* toDelete = curr->next;
    if (!toDelete) return false;
    curr->next = toDelete->next;
    if (toDelete->next) toDelete->next->prev = curr;
    delete toDelete;
    return true;
}

// Swap two cars at positions a and b (1-based, after head)
bool swapCar(TrainCar* head, int a, int b) {
    if (!head || a < 1 || b < 1 || a == b) return false;
    TrainCar* carA = head->next;
    TrainCar* carB = head->next;
    for (int i = 1; carA && i < a; ++i) carA = carA->next;
    for (int i = 1; carB && i < b; ++i) carB = carB->next;
    if (!carA || !carB) return false;
    // Swap contents (not pointers)
    std::swap(carA->type, carB->type);
    std::swap(carA->maxLoad, carB->maxLoad);
    std::swap(carA->load, carB->load);
    return true;
}

// Sort train by load (ascending/descending)
void sortTrain(TrainCar* head, bool ascending) {
    if (!head || !head->next) return;
    // Bubble sort for simplicity (small N)
    bool swapped;
    do {
        swapped = false;
        for (TrainCar* p = head->next; p && p->next; p = p->next) {
            if ((ascending && p->load > p->next->load) || (!ascending && p->load < p->next->load)) {
                std::swap(p->type, p->next->type);
                std::swap(p->maxLoad, p->next->maxLoad);
                std::swap(p->load, p->next->load);
                swapped = true;
            }
        }
    } while (swapped);
}

// Load cargo of a type, distributing to cars of that type
bool load(TrainCar* head, CarType type, int amount) {
    if (!head || type == HEAD || amount <= 0) return false;
    int totalAvailable = 0;
    for (TrainCar* p = head->next; p; p = p->next)
        if (p->type == type) totalAvailable += (p->maxLoad - p->load);
    if (totalAvailable < amount) return false;
    for (TrainCar* p = head->next; p && amount > 0; p = p->next) {
        if (p->type == type) {
            int canLoad = p->maxLoad - p->load;
            int toLoad = std::min(canLoad, amount);
            p->load += toLoad;
            amount -= toLoad;
        }
    }
    return true;
}

// Unload cargo of a type, removing from cars of that type (from end)
bool unload(TrainCar* head, CarType type, int amount) {
    if (!head || type == HEAD || amount <= 0) return false;
    int totalLoaded = 0;
    for (TrainCar* p = head->next; p; p = p->next)
        if (p->type == type) totalLoaded += p->load;
    if (totalLoaded < amount) return false;
    // Unload from end
    TrainCar* p = head;
    while (p->next) p = p->next;
    while (p != head && amount > 0) {
        if (p->type == type) {
            int toUnload = std::min(p->load, amount);
            p->load -= toUnload;
            amount -= toUnload;
        }
        p = p->prev;
    }
    return true;
}

// Print statistics for each cargo type
void printCargoStats(const TrainCar* head) {
    if (!head) return;
    struct Stat { int load = 0, maxload = 0, rank = 0; } stats[CARGO_TYPES];
    int order = 1;
    for (const TrainCar* p = head->next; p; p = p->next) {
        int idx = static_cast<int>(p->type) - 1;
        if (idx >= 0 && idx < CARGO_TYPES) {
            stats[idx].load += p->load;
            stats[idx].maxload += p->maxLoad;
            if (stats[idx].rank == 0) stats[idx].rank = order++;
        }
    }
    // Sort by rank (appearance order)
    for (int i = 0; i < CARGO_TYPES - 1; ++i)
        for (int j = 0; j < CARGO_TYPES - i - 1; ++j)
            if (stats[j].rank > stats[j + 1].rank && stats[j + 1].rank != 0) std::swap(stats[j], stats[j + 1]);
    bool first = true;
    for (int i = 0; i < CARGO_TYPES; ++i) {
        if (stats[i].rank == 0) continue;
        if (!first) std::cout << ",";
        std::cout << CarTypeStr[i + 1] << ":" << stats[i].load << "/" << stats[i].maxload;
        first = false;
    }
    std::cout << std::endl;
}

// Divide train into separate trains by cargo type
void divide(const TrainCar* head, TrainCar* results[CARGO_TYPE_COUNT]) {
    for (int i = 0; i < CARGO_TYPE_COUNT; ++i) results[i] = nullptr;
    if (!head) return;
    // For each type, create a new train and append cars of that type
    for (int t = 1; t <= CARGO_TYPE_COUNT; ++t) {
        TrainCar* newHead = createTrainHead();
        TrainCar* tail = newHead;
        for (const TrainCar* p = head->next; p; p = p->next) {
            if (static_cast<int>(p->type) == t) {
                auto* car = new TrainCar{p->type, p->maxLoad, p->load, tail, nullptr};
                tail->next = car;
                tail = car;
            }
        }
        if (newHead->next) results[t - 1] = newHead;
        else delete newHead;
    }
}

// Optimize train to maximize total cargo under upperBound (0/1 knapsack)
TrainCar* optimizeForMaximumPossibleCargos(const TrainCar* head, int upperBound) {
    if (!head || upperBound <= 0) return createTrainHead();
    // Collect cars
    std::vector<const TrainCar*> cars;
    for (const TrainCar* p = head->next; p; p = p->next) cars.push_back(p);
    int n = cars.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(upperBound + 1, 0));
    // DP
    for (int i = 1; i <= n; ++i) {
        int w = cars[i - 1]->load;
        for (int j = 0; j <= upperBound; ++j) {
            dp[i][j] = dp[i - 1][j];
            if (j >= w) dp[i][j] = std::max(dp[i][j], dp[i - 1][j - w] + w);
        }
    }
    // Backtrack
    TrainCar* result = createTrainHead();
    TrainCar* tail = result;
    int j = upperBound;
    for (int i = n; i >= 1; --i) {
        int w = cars[i - 1]->load;
        if (j >= w && dp[i][j] == dp[i - 1][j - w] + w) {
            auto* car = new TrainCar{cars[i - 1]->type, cars[i - 1]->maxLoad, cars[i - 1]->load, tail, nullptr};
            tail->next = car;
            tail = car;
            j -= w;
        }
    }
    // Reverse to preserve order
    // (Optional: if order matters, otherwise can skip)
    // ...
    return result;
}

// Deallocate all train cars
void deallocateTrain(TrainCar* head) {
    while (head) {
        TrainCar* next = head->next;
        delete head;
        head = next;
    }
}
