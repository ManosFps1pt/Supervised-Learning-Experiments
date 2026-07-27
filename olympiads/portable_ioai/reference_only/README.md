# Reference-only notebook snapshots

These notebooks preserve locally modified or untracked work that was found
inside the ignored source repositories during the portability export.

They are deliberately **not** part of the six-notebook runnable guarantee.
Their source repository, commit, original sparse path, working-tree status,
size, and SHA-256 are recorded in `reference_catalog.json`. Use them for
retrieval or comparison; move a notebook into the canonical `tasks/` workflow
and add a verified bootstrap contract before relying on it on another machine.
