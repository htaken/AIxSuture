<claude-mem-context>
# Memory Context

# [AIxSuture] recent context, 2026-04-20 6:08pm GMT+9

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 11 obs (5,342t read) | 222,940t work | 98% savings

### Apr 20, 2026
118 3:10p ⚖️ AIxSuture Phase 0 + Phase 1 Integration Plan Reviewed for SAM 3.1
119 3:11p 🔵 SAM 3.1 Image API: set_image Caches Backbone Embeddings; set_text_prompt Overwrites Previous Prompt
120 " 🔵 Phase 2/3 Verification Scripts Use Video Predictor API, Not Image API
121 " 🔵 AIxSuture Repo State: sam3 Submodule Has Uncommitted Changes; Datasets and tmp/ Are Untracked
122 " 🔵 Official Notebook Confirms reset_all_prompts() Required Before Each set_text_prompt() Call
123 " 🔵 build_sam3_image_model Downloads SAM3 (Not SAM3.1) Checkpoint by Default; batched_grounding_batch_size=1 Confirmed in Source
124 " 🔵 Dataset Contains 314 Surgical Videos Across 11 Packages; A31H.mp4 Confirmed Present
125 " 🔵 client_sam3.py Reveals Canonical Image API Usage: state Keys Are "boxes", "masks", "scores" Not Dict Output
126 3:13p 🔵 ffprobe Not Installed; cv2 Not Available in System Python; aixsuture/ Package Does Not Yet Exist
127 " 🔵 qualitative_test.py extract_frames Pattern: cv2.VideoCapture Dumps All Frames as JPEG to Directory
128 " 🔵 Sparse Frame Extraction Reference: get_frame_from_video() with av Fallback; verify_sam3p1_phase3.py Not at Repo Root

Access 223k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>