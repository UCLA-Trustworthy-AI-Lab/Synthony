# Visual Architecture Documentation (v1)

**Date:** 2026-02-08
**Version:** 1.0

Meaningful simplification of the **Synthony** architecture and **Recommender Engine** logic.

---

## 1. Abstract Architecture (High-Level)

This view shows how the three core packages interact to transform raw data into a model recommendation.

```mermaid
graph TD
    %% Styling
    classDef client fill:#f9f,stroke:#333,stroke-width:2px;
    classDef api fill:#bbf,stroke:#333,stroke-width:2px;
    classDef engine fill:#dfd,stroke:#333,stroke-width:2px;
    classDef db fill:#ffd,stroke:#333,stroke-width:2px;

    User((User)) -->|Uploads CSV/Parquet| API[API Gateway]
    User -->|Defines Intent| API
    
    subgraph "Synthony System"
        API:::api
        
        subgraph "Package 1: Data Infrastructure"
            Profiler[Stochastic Data Analyzer]:::engine
            ColumnProfiler[Column Analyzer]:::engine
        end
        
        subgraph "Package 3: Orchestration"
            RecEngine[Recommender Engine]:::engine
            Scorer[Hybrid Scorer]:::engine
        end
        
        subgraph "Package 2: Model Core"
            Registry[Model Capabilities JSON]:::db
        end
    end

    API -->|Raw Data| Profiler
    Profiler -->|Stress Profile| RecEngine
    ColumnProfiler -->|Column Analysis| RecEngine
    
    RecEngine -->|Reads Capabilities| Registry
    RecEngine -->|Calculates Scores| Scorer
    
    Scorer -->|Ranked Recommendation| API
    API -->|JSON Response| User

    %% Offline Loop
    subgraph "Offline Feedback Loop"
        Benchmarks[Benchmark Runner]
        Benchmarks -->|Update Scores| Registry
    end
```

---

## 2. Recommender Engine Logic (Detailed)

This flowchart details the decision process inside the `ModelRecommendationEngine`.

```mermaid
graph TD
    Start([Start Suggestion]) --> Input[/Input: Dataset Profile + Constraints/]
    
    %% Hard Filters
    subgraph "Step 1: Hard Filters"
        CheckGPU{GPU Available?}
        CheckPrivacy{Strict DP?}
        CheckRows{Rows > 50k?}
        
        Input --> CheckGPU
        CheckGPU -- Yes --> FilterGPU[Keep All Models]
        CheckGPU -- No --> FilterCPU[Filter: requires_gpu == true]
        
        FilterGPU --> CheckPrivacy
        FilterCPU --> CheckPrivacy
        
        CheckPrivacy -- Yes --> FilterDP[Keep Only: privacy_dp >= 3]
        CheckPrivacy -- No --> FilterNonDP[Keep All]
        
        FilterDP --> CheckRows
        FilterNonDP --> CheckRows
    end
    
    %% Stress Profiling
    subgraph "Step 2: Stress Profiling"
        CalcStress[Calculate Stress Scores]
        CalcStress --> Skew[Skew > 2.0?]
        CalcStress --> Card[Cardinality > 500?]
        CalcStress --> Zipf[Zipfian > 0.80?]
    end
    
    CheckRows --> CalcStress
    
    %% Scoring
    subgraph "Step 3: Scoring"
        Match{Match Capabilities?}
        Score[Calculate Weighted Score]
        
        Skew --> Match
        Card --> Match
        Zipf --> Match
        
        Match -->|Input: Stress Vector| Score
        Registry[(Model Registry)] -->|Input: Capability Scores| Score
    end
    
    Score --> RankedList[Initial Ranked List]
    
    %% Tie Breaking
    subgraph "Step 4: Tie-Breaking"
        TieCheck{Top Scores Close?}
        SmallData{Rows < 1000?}
        Speed{Intent = Latency?}
        
        RankedList --> TieCheck
        TieCheck -- No --> FinalResult
        TieCheck -- Yes --> SmallData
        
        SmallData -- Yes --> BoostTree[Boost: ARF, CART]
        SmallData -- No --> Speed
        
        Speed -- Yes --> BoostFast[Boost: CART, TVAE]
        Speed -- No --> Quality[Boost: TabDDPM, TabSyn]
        
        BoostTree --> FinalResult
        BoostFast --> FinalResult
        Quality --> FinalResult
    end
    
    FinalResult([Final Recommendation])
```


What are the stree profiles DP, tiebreaker, 