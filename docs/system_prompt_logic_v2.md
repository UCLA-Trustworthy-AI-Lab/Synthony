# System Prompt Logic v2

```mermaid
flowchart TD
    Start([Start: Profile Data]) --> Filter{Hard Filters}
    
    %% Filter Phase
    Filter -- "CPU Only?" --> RemoveGPU[Remove: TabDDPM, TabSyn, GReaT]
    Filter -- "Strict DP?" --> RemoveNonDP[Keep Only: PATECTGAN, DPCART, AIM]
    RemoveGPU --> Score
    RemoveNonDP --> Score
    Filter -- "None" --> Score[Calculate Weighted Capability Scores]

    %% Scoring Phase
    Score --> HardProb{Is Hard Problem?\\n(Skew>2 & Card>500 & Zipf>0.80)}
    
    %% The Hard Problem Logic with Safety Check
    HardProb -- Yes --> CheckSize{Rows > 50k?}
    CheckSize -- Yes --> WarnSlow[Recommend: TabDDPM\\n(GReaT too slow for size)]
    CheckSize -- No --> CheckCand{Is GReaT in\\nCandidate Pool?}
    
    CheckCand -- Yes --> RecGreat[Recommend: GReaT\\n(Best for Complex Tail)]
    CheckCand -- No --> RecTabSyn[Recommend: TabSyn/ARF\\n(Best available backup)]
    
    %% Normal Flow
    HardProb -- No --> Top2{Top 2 within 5%?}
    Top2 -- No --> Winner[Recommend Highest Score]
    
    %% Tie Breaking
    Top2 -- Yes --> SmallData{Rows < 1000?}
    SmallData -- Yes --> RecARF[Recommend: ARF\\n(Prevents Overfitting)]
    SmallData -- No --> Speed{Prefer Speed?}
    Speed -- Yes --> RecTVAE[Recommend: TVAE/CTGAN]
    Speed -- No --> RecDiff[Recommend: TabDDPM]
    
    %% Outputs
    WarnSlow --> Output
    RecGreat --> Output
    RecTabSyn --> Output
    Winner --> Output
    RecARF --> Output
    RecTVAE --> Output
    RecDiff --> Output
```
