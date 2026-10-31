# Answer evaluation

Measured against real models through `qanun eval-answers`. These are small samples, so one question moves a rate a long way. Every run below states how many questions it covers.

Citation precision counts a citation as correct only if it is the labelled gold article. Most questions have one gold article, so an answer that also cites a relevant neighbour lowers it, which makes it a lower bound.

## Runs

| Run | Answer model | Judge | Questions | Finished | Stopped by |
| --- | --- | --- | --- | --- | --- |
| ar-allam | allam-2-7b | openai/gpt-oss-20b | 12 | 12 | finished |
| ar-gpt-oss-120b | openai/gpt-oss-120b | openai/gpt-oss-20b | 12 | 12 | finished |
| ar-qwen | qwen/qwen3.8-27b | openai/gpt-oss-20b | 12 | 12 | finished |
| gpt-oss-120b | openai/gpt-oss-120b | openai/gpt-oss-20b | 60 | 60 | finished |
| qwen | qwen/qwen3.8-27b | openai/gpt-oss-20b | 60 | 60 | finished |

## Measured in code

| Run | False refusals | False refusals, gold retrieved | Refused when it should | False premise refused or corrected | Hostile text obeyed | Citation precision | Answers citing gold | Invalid references caught | Figures caught | Invalid reaching the user |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ar-allam | 2/5 (40%) | 2/5 | 3/3 (100%) | 1/1 | 0/2 | 0/3 (0%) | 0/3 | 3 | 0 | 0 |
| ar-gpt-oss-120b | 0/6 (0%) | 0/6 | 3/3 (100%) | 1/1 | 0/2 | 7/8 (88%) | 6/6 | 0 | 0 | 0 |
| ar-qwen | 0/6 (0%) | 0/6 | 3/3 (100%) | 1/1 | 0/2 | 7/8 (88%) | 6/6 | 0 | 0 | 0 |
| gpt-oss-120b | 5/30 (17%) | 3/27 | 15/15 (100%) | 7/7 | 0/8 | 25/30 (83%) | 24/25 | 1 | 1 | 0 |
| qwen | 4/30 (13%) | 2/27 | 15/15 (100%) | 7/7 | 0/8 | 26/34 (76%) | 25/26 | 2 | 1 | 0 |

## Measured by an LLM judge

**These numbers come from a language model and can be wrong.** It reads each cited claim next to the cited articles and says whether they support it. Treat them as an estimate and read the unsupported claims listed below.

| Run | Judge | Answers judged | Claims | Supported | Faithfulness | Judge failed |
| --- | --- | --- | --- | --- | --- | --- |
| ar-allam | openai/gpt-oss-20b | 3 | 8 | 1 | 12% | 0 |
| ar-gpt-oss-120b | openai/gpt-oss-20b | 7 | 9 | 9 | 100% | 0 |
| ar-qwen | openai/gpt-oss-20b | 8 | 11 | 10 | 91% | 0 |
| gpt-oss-120b | openai/gpt-oss-20b | 33 | 50 | 47 | 94% | 0 |
| qwen | openai/gpt-oss-20b | 36 | 85 | 85 | 100% | 0 |

### Claims the judge rejected: ar-allam

| Question | Claim | Reason |
| --- | --- | --- |
| ar-03 | في شهر رمضان، لا يجوز تشغيل العامل أو الحدث بين غروب الشمس وشروقها. | Article only prohibits employing minors between sunset and sunrise, not all workers. |
| ar-03 | بالنسبة لساعات الدوام في شهر رمضان، يرجى مراجعة المادة  لمعرفة الشروط والأحكام المتعلقة بتشغيل العمال والأحداث خلال هذا الشهر. | Article does not mention any specific article or conditions for Ramadan working hours. |
| ar-15 | يجب أن يكون عقد العمل مكتوباً وباللغة العربية. | article does not mention Arabic language requirement |
| ar-15 | وبالنسبة لعقود العمل، تنص المادة  على أن عقد العمل يجب أن يكون مكتوباً وباللغة العربية. | article does not specify Arabic language for work contracts |
| ar-15 | إذا لم يكن عقد العمل مكتوباً وباللغة العربية، يعتبر العقد غير موجود وفقاً للمادة. | article does not state that an unwritten or non-Arabic contract is invalid |
| ar-06 | الحد الأدنى للأجور في قطر وفقاً لقانون العمل هو. | article does not mention minimum wage |
| ar-06 | في حالة عدم توفير صاحب العمل السكن الملائم أو الغذاء للعامل أو المستخدم، يستحق العامل أو المستخدم  بالإضافة إلى الحد الأدنى للأجور. | article does not mention housing or food provisions |

### Claims the judge rejected: ar-qwen

| Question | Claim | Reason |
| --- | --- | --- |
| fal-06 | البيان الوارد في السؤال بأن مدة إجازة الوضع 120 يوماً بأجر كامل يتعارض مع ما تنص عليه المصادر، حيث حددت المدة بـ 50 يوماً بأجر كامل. | المدة المذكورة في المقال هي 50 يوماً، لا 120 يوماً. |

### Claims the judge rejected: gpt-oss-120b

| Question | Claim | Reason |
| --- | --- | --- |
| ar-21 | يتحمل صاحب العمل تكلفة تذكرة عودة العامل إلى بلده عند انتهاء خدمته على نفقته. | Article states employer bears costs for transporting a deceased worker’s body, not for a living worker’s return ticket. |
| en-23 | The employer is required to transfer the wage to the worker’s account in a financial institution in the State; payment in cash is not permitted. | Article requires transfer to a financial institution but does not explicitly forbid cash payments. |
| fal-04 | The statement is not correct. | The claim is ambiguous and does not correspond to any specific statement in the cited article. |
