# Historical CI evidence for Strawberry Fields

This record preserves the identity, report outcomes, byte counts and SHA-256
digests of 84 original GitHub Actions ZIPs from eight runs associated with
[#46](https://github.com/KharonByte/strawberryfields/pull/46),
[#47](https://github.com/KharonByte/strawberryfields/pull/47) and
[#49](https://github.com/KharonByte/strawberryfields/pull/49).
The [artifact index](ci-evidence.csv) was checked against the original ZIPs.
This record includes no raw ZIPs or logs.

| Execution | Actual checkout | Tests ZIPs | Scoped ZIPs |
| --- | --- | ---: | ---: |
| #46 exact merge, predecessor for #47 | `cb91f446f0f97d43ffa5d7aafed916a4581cd3e7` | [15](https://github.com/KharonByte/strawberryfields/actions/runs/43215567604) | Not indexed |
| #47 PR test merge | `8d9a1b751fc3df073ba1f4ced525004b148798da` | [15](https://github.com/KharonByte/strawberryfields/actions/runs/43241068177) | [3](https://github.com/KharonByte/strawberryfields/actions/runs/43241068469) |
| #47 exact merge | `7020d7a901b1705a16073c24974148b693f7ff3b` | [15](https://github.com/KharonByte/strawberryfields/actions/runs/43340149614) | [3](https://github.com/KharonByte/strawberryfields/actions/runs/43340149713) |
| #49 earlier PR test merge | `b5c05414d113260f6d8c2040e4ed55ca5c5db47f` | [15](https://github.com/KharonByte/strawberryfields/actions/runs/43478175614) | Not indexed |
| #49 later PR test merge | `df24c8084d02d0ad9ba2f0db42b6b5205fba3cbb` | [15](https://github.com/KharonByte/strawberryfields/actions/runs/43514124598) | [3](https://github.com/KharonByte/strawberryfields/actions/runs/43514124462) |

## What the records establish

- Compared with #46 exact merge, #47 exact merge adds 164 passing JUnit rows
  per all-backends profile, 140 per non-TensorFlow and local-without-cloud
  profile, and 12 per TensorFlow batch pure and mixed profile. These counts
  hold on each of the three tested operating systems. No existing JUnit
  result row disappeared or changed outcome under the comparison key:
  classname, name, and failure/error/skipped tag and type. Profiles overlap;
  their counts are not a total of unique tests.
- The earlier #49 Tests run has 14 ZIPs produced on attempt 1 and one on
  attempt 2. The attempt-2 job listing includes inherited results; it does
  not show that every job executed again. The index records the producing
  attempt and job rather than the latest listing's job alias.
- The CSV outcome fields count reported JUnit rows. It separates expected
  failures from ordinary skips. Deselected cases do not appear in JUnit.
  Scoped ZIP counts combine six reports, which may overlap. A zero in the
  failure or error column applies to the recorded reports, not every job
  that was scheduled.

## Verifying a copy

Match the original Actions ZIP to its artifact ID, byte count and SHA-256 in
the index. Use the run and producer job to find the execution context. GitHub
Actions links identify the run and can outlive the downloadable ZIP; a
checksum does not make an expired archive recoverable.

This is historical CI test evidence. It does not replace independent
scientific validation, certify numerical accuracy or establish current
support. The historical #49 PR runs above are not the final merged result.
Documentation builds, full workflow logs and independent scientific
evidence are outside this record.
