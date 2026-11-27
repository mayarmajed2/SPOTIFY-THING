# Track Matching & Entity Resolution (`docs/ENTITY_RESOLUTION.md`)

**Implementation:** `extract_version_tag`, `normalize_exact`, `normalize_packaging_stripped` in [backend/data_pipeline/build_verified_dataset.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/data_pipeline/build_verified_dataset.py)

---

## 1. Problem in Legacy Pipeline
The legacy pipeline stripped `(live)`, `(remaster)`, `(radio edit)`, `(version)`, `(deluxe)`, and `(bonus)` unconditionally and collapsed duplicate `(artist_norm, track_norm)` keys via `drop_duplicates(keep="first")`, causing live/acoustic/remix recordings to collide with studio tracks (`3,261` collisions in the legacy 40k catalog).

## 2. Version-Aware Matching Strategy (v2)
1. **Semantic Version Extraction (`version_tag`):** Words indicating a distinct musical recording (`live`, `remix`, `acoustic`, `unplugged`, `demo`, `instrumental`, `karaoke`, `a cappella`, `orchestral`, `dub`, `club mix`) are extracted into a canonical `version_tag` (`"studio"` if none). Two tracks with different `version_tag`s can **never** match.
2. **Packaging Suffix Stripping:** Only non-semantic release suffixes (`feat.`, `ft.`, `remaster(ed)`, `deluxe`, `bonus track`, `explicit`, `clean`, `album version`, `single version`, `YYYY remaster`) are stripped for Tier 2 fallback matching.
3. **Uniqueness Enforcement:** Any normalized lookup key mapping to `> 1` distinct `spotify_track_id` in the candidate catalog is classified as **`ambiguous`** and excluded from training/evaluation interactions rather than arbitrarily picking one.

## 3. Match Confidence Tiers & Empirical Results (16,936,134 Raw Last.fm Events)

| Match Tier | Confidence | Used in Training/Eval? | Empirical Event Count | Share of Raw Events |
| :--- | :---: | :---: | ---: | ---: |
| `exact_normalized` | `1.00` | Yes | 1,251,638 | 7.39% |
| `packaging_stripped_high` | `0.85` | Yes | 169,591 | 1.00% |
| `ambiguous` (multi-target collision) | `0.40` | **No (Excluded)** | 502,528 | 2.97% |
| `unmatched` | `0.00` | **No (Excluded)** | 15,012,195 | 88.64% |
| **Total Reliable Matches** | **≥ 0.85** | **Yes** | **1,421,229** | **8.39%** |

## 4. Known Limitations
- Last.fm 1K lacks Spotify IDs, so matching relies on artist + track text + version tag without album/duration verification.
- The 88.64% unmatched rate reflects the mismatch between Last.fm 2005–2009 long-tail catalog and the 10-slice Spotify MPD + 114k Spotify CSV snapshot. Expanding to more MPD slices or MusicBrainz-to-ISRC mapping would increase coverage.
