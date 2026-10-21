"use client";



import { useCallback, useEffect, useState, type FormEvent } from "react";



const API = "http://127.0.0.1:8000";



type RecordItem = Record<string, unknown>;




type JobForm = {
  model_id: string;
  dataset_id: string;
  method: string;
  epochs: number;
  learning_rate: number;
  batch_size: number;
  description: string;
  compute_target: "local" | "amd_mi300x";
};




const endpoints = [

  { key: "models", label: "Models", path: "/api/models/" },

  { key: "inferenceHistory", label: "Inference History", path: "/api/inference-history/?limit=50" },

  { key: "experiments", label: "Experiments", path: "/api/experiments/?limit=50" },

  { key: "datasets", label: "Datasets", path: "/api/datasets/" },

  { key: "jobs", label: "Training Jobs", path: "/api/training-jobs/" },

  { key: "artifacts", label: "Artifacts", path: "/api/artifacts/" },

  { key: "evaluations", label: "Evaluations", path: "/api/evaluations/" },

] as const;



function getText(value: unknown, fallback = "—") {

  return typeof value === "string" || typeof value === "number"

    ? String(value)

    : fallback;

}



function statusClass(status: string) {

  if (status === "completed") return "status completed";

  if (status === "failed") return "status failed";

  if (status === "running") return "status running";

  return "status queued";

}



export default function Dashboard() {

  const [data, setData] = useState<Record<string, RecordItem[]>>({});

  const [loading, setLoading] = useState(true);

  const [error, setError] = useState("");

  const [lastUpdated, setLastUpdated] = useState("");

  const [inferenceArtifactId, setInferenceArtifactId] = useState("");

  const [inferencePrompt, setInferencePrompt] = useState("");

  const [maxNewTokens, setMaxNewTokens] = useState(128);

  const [temperature, setTemperature] = useState(0.7);

  const [generating, setGenerating] = useState(false);

  const [generatedResponse, setGeneratedResponse] = useState("");

  const [inferenceError, setInferenceError] = useState("");

  const [comparing, setComparing] = useState(false);

  const [comparisonError, setComparisonError] = useState("");

  const [comparisonResult, setComparisonResult] = useState<RecordItem | null>(null);





  const [jobForm, setJobForm] = useState<JobForm>({

    model_id: "",

    dataset_id: "",

    method: "lora",

    epochs: 1,

    learning_rate: 0.0002,

    batch_size: 1,

    description: "",

    compute_target: "amd_mi300x",

  });



  const [creatingJob, setCreatingJob] = useState(false);

  const [jobMessage, setJobMessage] = useState("");

  const [jobError, setJobError] = useState("");



  const refresh = useCallback(async () => {

    try {

      const results = await Promise.all(

        endpoints.map(async (endpoint) => {

          const response = await fetch(`${API}${endpoint.path}`, {

            cache: "no-store",

          });



          if (!response.ok) {

            throw new Error(

              `${endpoint.label} API returned HTTP ${response.status}`

            );

          }



          const json: unknown = await response.json();



          if (!Array.isArray(json)) {

            throw new Error(`${endpoint.label} API did not return a list`);

          }



          return [endpoint.key, json as RecordItem[]] as const;

        })

      );



      setData(Object.fromEntries(results));

      setError("");

      setLastUpdated(new Date().toLocaleTimeString());

    } catch (err) {

      setError(

        err instanceof Error ? err.message : "Could not connect to the API."

      );

    } finally {

      setLoading(false);

    }

  }, []);



  const handleGenerate = async (event: FormEvent<HTMLFormElement>) => {

  event.preventDefault();

  setGenerating(true);

  setGeneratedResponse("");

  setInferenceError("");



  try {

    const response = await fetch(`${API}/api/inference/generate`, {

      method: "POST",

      headers: { "Content-Type": "application/json" },

      body: JSON.stringify({

        artifact_id: inferenceArtifactId,

        prompt: inferencePrompt,

        max_new_tokens: maxNewTokens,

        temperature,

      }),

    });



    const result = await response.json();



    if (!response.ok) {

      throw new Error(result.detail || `HTTP ${response.status}`);

    }



    const responseText = typeof result.response === "string" ? result.response : "";
    setGeneratedResponse(responseText);

    // Save the successful generation without turning a save failure into a generation failure.
    try {
      const historyResponse = await fetch(`${API}/api/inference-history/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          artifact_id: inferenceArtifactId,
          prompt: inferencePrompt,
          response: responseText,
          max_new_tokens: maxNewTokens,
          temperature,
          generation_time_seconds: Number(result.generation_time_seconds ?? 0),
          prompt_tokens: Number(result.prompt_tokens ?? 0),
          generated_tokens: Number(result.generated_tokens ?? 0),
        }),
      });
      if (historyResponse.ok) {
        await refresh();
      } else {
        console.error("Generation succeeded, but inference history was not saved.");
      }
    } catch (saveError) {
      console.error("Generation succeeded, but inference history could not be saved.", saveError);
    }

  } catch (err) {

    setInferenceError(

      err instanceof Error ? err.message : "Generation failed."

    );

  } finally {

    setGenerating(false);

  }

};





const handleCompare = async (event: FormEvent<HTMLFormElement>) => {

  event.preventDefault();



  setComparing(true);

  setComparisonError("");

  setComparisonResult(null);



  try {

    const response = await fetch(`${API}/api/inference/compare`, {

      method: "POST",

      headers: { "Content-Type": "application/json" },

      body: JSON.stringify({

        artifact_id: inferenceArtifactId,

        prompt: inferencePrompt,

        max_new_tokens: maxNewTokens,

        temperature: 0,

      }),

    });



    const result: unknown = await response.json();



    if (!response.ok) {

      const detail =

        typeof result === "object" &&

        result !== null &&

        "detail" in result

          ? result.detail

          : undefined;



      throw new Error(

        typeof detail === "string"

          ? detail

          : `Comparison failed with HTTP ${response.status}`

      );

    }



    const comparison = result as RecordItem;
    setComparisonResult(comparison);

    const baseRaw = comparison.base_model;
    const tunedRaw = comparison.fine_tuned_model;
    const base = typeof baseRaw === "object" && baseRaw !== null ? baseRaw as RecordItem : {};
    const tuned = typeof tunedRaw === "object" && tunedRaw !== null ? tunedRaw as RecordItem : {};

    // Persist the comparison for later review.
    try {
      const experimentResponse = await fetch(`${API}/api/experiments/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          artifact_id: inferenceArtifactId,
          prompt: inferencePrompt,
          max_new_tokens: maxNewTokens,
          temperature: 0,
          base_model_name: getText(base.name, "Base model"),
          fine_tuned_model_name: getText(tuned.name, "Fine-tuned model"),
          base_response: getText(base.response, ""),
          fine_tuned_response: getText(tuned.response, ""),
          base_generation_time_seconds: Number(base.generation_time_seconds ?? 0),
          fine_tuned_generation_time_seconds: Number(tuned.generation_time_seconds ?? 0),
          base_prompt_tokens: Number(base.prompt_tokens ?? 0),
          fine_tuned_prompt_tokens: Number(tuned.prompt_tokens ?? 0),
          base_generated_tokens: Number(base.generated_tokens ?? 0),
          fine_tuned_generated_tokens: Number(tuned.generated_tokens ?? 0),
        }),
      });
      if (experimentResponse.ok) {
        await refresh();
      } else {
        console.error("Comparison succeeded, but the experiment was not saved.");
      }
    } catch (saveError) {
      console.error("Comparison succeeded, but the experiment could not be saved.", saveError);
    }

  } catch (err) {

    setComparisonError(

      err instanceof Error ? err.message : "Comparison failed."

    );

  } finally {

    setComparing(false);

  }

};





  const handleCreateJob = async (event: FormEvent<HTMLFormElement>) => {

    event.preventDefault();



    setCreatingJob(true);

    setJobMessage("");

    setJobError("");



    try {

      const response = await fetch(`${API}/api/training-jobs/`, {

        method: "POST",

        headers: {

          "Content-Type": "application/json",

        },

        body: JSON.stringify(jobForm),

      });



      const result: unknown = await response.json();



      if (!response.ok) {

        const detail =

          typeof result === "object" &&

          result !== null &&

          "detail" in result

            ? result.detail

            : undefined;



        throw new Error(

          typeof detail === "string"

            ? detail

            : `Request failed with HTTP ${response.status}`

        );

      }



      const createdId =

        typeof result === "object" && result !== null && "id" in result

          ? getText(result.id)

          : "created";



      setJobMessage(`Training job created successfully: ${createdId}`);



      setJobForm((previous) => ({

        ...previous,

        description: "",

      }));



      await refresh();

    } catch (err) {

      setJobError(

        err instanceof Error

          ? err.message

          : "Could not create the training job."

      );

    } finally {

      setCreatingJob(false);

    }

  };



  
useEffect(() => {
  void refresh();

  const timer = window.setInterval(() => {
    void refresh();
  }, 2000);

  return () => window.clearInterval(timer);
}, [refresh]);




  const models = data.models ?? [];

  const datasets = data.datasets ?? [];

  const jobs = data.jobs ?? [];

  const artifacts = data.artifacts ?? [];

  const evaluations = data.evaluations ?? [];

  const inferenceHistory = data.inferenceHistory ?? [];

  const experiments = data.experiments ?? [];



  const completed = jobs.filter((job) => job.status === "completed").length;

  const running = jobs.filter((job) => job.status === "running").length;

  const failed = jobs.filter((job) => job.status === "failed").length;



  const cards = [

    { label: "Registered Models", value: models.length, icon: "◈" },

    { label: "Datasets", value: datasets.length, icon: "▤" },

    { label: "Training Jobs", value: jobs.length, icon: "⌘" },

    { label: "Saved Adapters", value: artifacts.length, icon: "⬡" },

  ];



  return (

    <main className="app-shell">

      <aside className="sidebar">

        <a className="brand" href="#overview">

          <span className="brand-icon">F</span>

          <span>

            Forge<span className="brand-light">ML</span>

          </span>

        </a>



        <a className="nav-item" href="#playground">

          ✦ <span>Inference Playground</span>

        </a>



        <p className="nav-heading">WORKSPACE</p>

        <a className="nav-item active" href="#overview">

          ▦ <span>Overview</span>

        </a>

        <a className="nav-item" href="#training">

          ⌘ <span>Training Jobs</span>

        </a>

        <a className="nav-item" href="#models">

          ◈ <span>Models</span>

        </a>

        <a className="nav-item" href="#datasets">

          ▤ <span>Datasets</span>

        </a>

        <a className="nav-item" href="#artifacts">

          ⬡ <span>Artifacts</span>

        </a>

        <a className="nav-item" href="#evaluations">

          ◎ <span>Evaluations</span>

        </a>

        <a className="nav-item" href="#inference-history">

          ↺ <span>Inference History</span>

        </a>

        <a className="nav-item" href="#experiments">

          ⇄ <span>Experiments</span>

        </a>

        <a className="nav-item" href="#experiment-lab">
          ✦ <span>Experiment Lab</span>
        </a>



        <div className="sidebar-bottom">

          <span className="online-dot" />

          <div>

            <strong>Local environment</strong>

            <small>FastAPI · localhost:8000</small>

          </div>

        </div>

      </aside>



      <section className="main-area">

        <header className="topbar">

          <div>

            <p className="eyebrow">WORKSPACE / OVERVIEW</p>

            <h1>Dashboard</h1>

          </div>

          <button

            className="refresh-button"

            onClick={() => void refresh()}

            disabled={loading}

          >

            ↻ &nbsp; {loading ? "Refreshing..." : "Refresh data"}

          </button>

        </header>



        {error && (

          <div className="error-banner">

            <strong>Backend connection issue</strong>

            <span>{error}</span>

            <small>

              Check that FastAPI is running at {API} and that CORS allows

              localhost:3000.

            </small>

          </div>

        )}



        <section id="overview" className="welcome-panel">

          <div>

            <p className="eyebrow accent">MODEL DEVELOPMENT WORKSPACE</p>

            <h2>Build, fine-tune, and evaluate models.</h2>

            <p>

              Your training infrastructure at a glance. Data below is fetched

              from your ForgeML backend.

            </p>

          </div>

          <div className="welcome-symbol">

            F<span>.</span>

          </div>

        </section>



        <div className="section-heading">

          <div>

            <h2>Workspace metrics</h2>

            <p>Live counts from your backend APIs</p>

          </div>

          <span className="live-label">

            <span className="online-dot" /> LIVE

          </span>

        </div>



        <section className="metrics-grid">

          {cards.map((card) => (

            <article className="metric-card" key={card.label}>

              <div className="metric-top">

                <span>{card.label}</span>

                <span className="metric-icon">{card.icon}</span>

              </div>

              <strong className="metric-value">

                {loading ? "—" : card.value}

              </strong>

              <span className="metric-foot">From persisted API records</span>

            </article>

          ))}

        </section>



        <section className="status-grid">

          <article className="status-card">

            <span className="status-indicator green" />

            <div>

              <strong>{completed}</strong>

              <span>Completed jobs</span>

            </div>

          </article>

          <article className="status-card">

            <span className="status-indicator blue" />

            <div>

              <strong>{running}</strong>

              <span>Running jobs</span>

            </div>

          </article>

          <article className="status-card">

            <span className="status-indicator red" />

            <div>

              <strong>{failed}</strong>

              <span>Failed jobs</span>

            </div>

          </article>

          <article className="status-card">

            <span className="status-indicator purple" />

            <div>

              <strong>{evaluations.length}</strong>

              <span>Evaluations</span>

            </div>

          </article>

        </section>



        <section id="training" className="content-panel">

          <div className="create-job-panel">

            <h3>Create a training job</h3>

            <p>

              Select a registered model and dataset to configure a LoRA job.

            </p>



            <form onSubmit={handleCreateJob} className="job-form">

              <label>

                Base model

                <select

                  required

                  value={jobForm.model_id}

                  onChange={(event) =>

                    setJobForm({

                      ...jobForm,

                      model_id: event.target.value,

                    })

                  }

                >

                  <option value="">Select a model</option>

                  {models.map((model, index) => (

                    <option

                      key={getText(model.id, String(index))}

                      value={getText(model.id, "")}

                    >

                      {getText(model.name, getText(model.model_id))}

                    </option>

                  ))}

                </select>

              </label>



              <label>

                Dataset

                <select

                  required

                  value={jobForm.dataset_id}

                  onChange={(event) =>

                    setJobForm({

                      ...jobForm,

                      dataset_id: event.target.value,

                    })

                  }

                >

                  <option value="">Select a dataset</option>

                  {datasets.map((dataset, index) => (

                    <option

                      key={getText(dataset.id, String(index))}

                      value={getText(dataset.id, "")}

                    >

                      {getText(dataset.name)}

                    </option>

                  ))}

                </select>

              </label>

                  
              <label>
                Compute target
                <select
                  value={jobForm.compute_target}
                  onChange={(event) =>
                    setJobForm({
                      ...jobForm,
                      compute_target: event.target.value as "local" | "amd_mi300x",
                    })
                  }
                >
                  <option value="amd_mi300x">
                    AMD Instinct MI300X (GPU)
                  </option>
                  <option value="local">
                    Local machine (CPU/GPU)
                  </option>
                </select>
              </label>


              <label>

                Training method

                <select

                  value={jobForm.method}

                  onChange={(event) =>

                    setJobForm({

                      ...jobForm,

                      method: event.target.value,

                    })

                  }

                >

                  <option value="lora">LoRA fine-tuning</option>

                </select>

              </label>



              <div className="job-form-row">

                <label>

                  Epochs

                  <input

                    type="number"

                    min="1"

                    max="10"

                    required

                    value={jobForm.epochs}

                    onChange={(event) =>

                      setJobForm({

                        ...jobForm,

                        epochs: Number(event.target.value),

                      })

                    }

                  />

                </label>



                <label>

                  Batch size

                  <input

                    type="number"

                    min="1"

                    max="16"

                    required

                    value={jobForm.batch_size}

                    onChange={(event) =>

                      setJobForm({

                        ...jobForm,

                        batch_size: Number(event.target.value),

                      })

                    }

                  />

                </label>



                <label>

                  Learning rate

                  <input

                    type="number"

                    min="0.000001"

                    max="0.01"

                    step="any"

                    required

                    value={jobForm.learning_rate}

                    onChange={(event) =>

                      setJobForm({

                        ...jobForm,

                        learning_rate: Number(event.target.value),

                      })

                    }

                  />

                </label>

              </div>



              <label>

                Description (optional)

                <input

                  type="text"

                  maxLength={200}

                  value={jobForm.description}

                  placeholder="e.g. First LoRA experiment"

                  onChange={(event) =>

                    setJobForm({

                      ...jobForm,

                      description: event.target.value,

                    })

                  }

                />

              </label>



              {jobError && (

                <div className="error-banner" role="alert">

                  <strong>Could not create job</strong>

                  <span>{jobError}</span>

                </div>

              )}



              {jobMessage && (

                <div className="success-banner" role="status">

                  {jobMessage}

                </div>

              )}



              <button

                type="submit"

                className="refresh-button"

                disabled={

                  creatingJob ||

                  models.length === 0 ||

                  datasets.length === 0

                }

              >

                {creatingJob ? "Creating job..." : "Create training job"}

              </button>



              {(models.length === 0 || datasets.length === 0) && (

                <p className="muted">

                  Register at least one model and one dataset before creating

                  a job.

                </p>

              )}

            </form>

          </div>



          <div className="section-heading">

            <div>

              <h2>Recent training jobs</h2>

              <p>Latest jobs reported by the API</p>

            </div>

            <span className="count-label">{jobs.length} total</span>

          </div>



          {jobs.length === 0 ? (

            <div className="empty-state">

              <span>⌘</span>

              <strong>No training jobs yet</strong>

              <p>Create a job using the form above to see it here.</p>

            </div>

          ) : (

            <div className="table-wrap">

              <table>

                <thead>

                  <tr>

                    <th>JOB</th>

                    <th>METHOD</th>

                    <th>STATUS</th>

                    <th>PROGRESS</th>

                    <th>STAGE</th>

                  </tr>

                </thead>

                <tbody>

                  {[...jobs].reverse().slice(0, 8).map((job, index) => (

                    <tr key={getText(job.id, String(index))}>

                      <td>

                        <code>{getText(job.id).slice(0, 13)}…</code>

                      </td>

                      <td>{getText(job.method)}</td>

                      <td>

                        <span className={statusClass(getText(job.status))}>

                          {getText(job.status)}

                        </span>

                      </td>

                      <td>

                        <div className="progress-cell">

                          <div className="progress-track">

                            <div

                              style={{

                                width: `${Math.min(

                                  100,

                                  Math.max(0, Number(job.progress) || 0)

                                )}%`,

                              }}

                            />

                          </div>

                          <span>{getText(job.progress, "0")}%</span>

                        </div>

                      </td>

                      <td>{getText(job.current_stage)}</td>

                    </tr>

                  ))}

                </tbody>

              </table>

            </div>

          )}

        </section>



        <div className="lower-grid">

          <section id="models" className="content-panel">

            <div className="section-heading">

              <div>

                <h2>Models</h2>

                <p>Registered base models</p>

              </div>

              <span className="count-label">{models.length}</span>

            </div>

            {models.length === 0 ? (

              <p className="muted">No models registered.</p>

            ) : (

              models.slice(0, 5).map((model, index) => (

                <div

                  className="list-row"

                  key={getText(model.id, String(index))}

                >

                  <span className="list-icon">◈</span>

                  <div className="list-copy">

                    <strong>

                      {getText(model.name, getText(model.model_id))}

                    </strong>

                    <small>{getText(model.model_id)}</small>

                  </div>

                </div>

              ))

            )}

          </section>



          <section id="datasets" className="content-panel">

            <div className="section-heading">

              <div>

                <h2>Datasets</h2>

                <p>Available training data</p>

              </div>

              <span className="count-label">{datasets.length}</span>

            </div>

            {datasets.length === 0 ? (

              <p className="muted">No datasets registered.</p>

            ) : (

              datasets.slice(0, 5).map((dataset, index) => (

                <div

                  className="list-row"

                  key={getText(dataset.id, String(index))}

                >

                  <span className="list-icon">▤</span>

                  <div className="list-copy">

                    <strong>{getText(dataset.name)}</strong>

                    <small>

                      {getText(dataset.format)} · {getText(dataset.purpose)}

                    </small>

                  </div>

                </div>

              ))

            )}

          </section>

        </div>



        <div className="lower-grid">

          <section id="artifacts" className="content-panel">

            <div className="section-heading">

              <div>

                <h2>Saved artifacts</h2>

                <p>Registered adapters</p>

              </div>

              <span className="count-label">{artifacts.length}</span>

            </div>

            {artifacts.length === 0 ? (

              <p className="muted">No artifacts registered.</p>

            ) : (

              artifacts.slice(0, 5).map((artifact, index) => (

                <div

                  className="list-row"

                  key={getText(artifact.id, String(index))}

                >

                  <span className="list-icon">⬡</span>

                  <div className="list-copy">

                    <strong>{getText(artifact.type, "LoRA adapter")}</strong>

                    <small>{getText(artifact.training_job_id)}</small>

                    <small className="path-text">

                      {getText(artifact.path)}

                    </small>

                  </div>

                </div>

              ))

            )}

          </section>



          <section id="evaluations" className="content-panel">

            <div className="section-heading">

              <div>

                <h2>Evaluations</h2>

                <p>Model quality checks</p>

              </div>

              <span className="count-label">{evaluations.length}</span>

            </div>

            {evaluations.length === 0 ? (

              <p className="muted">No evaluations yet.</p>

            ) : (

              [...evaluations].reverse().slice(0, 5).map((evaluation, index) => (

                <div

                  className="list-row"

                  key={getText(evaluation.id, String(index))}

                >

                  <span className="list-icon">◎</span>

                  <div className="list-copy">

                    <strong>

                      {getText(evaluation.description, "Evaluation job")}

                    </strong>

                    <small>

                      {getText(evaluation.current_stage)} ·{" "}

                      {getText(evaluation.progress, "0")}%

                    </small>

                    <span

                      className={statusClass(getText(evaluation.status))}

                    >

                      {getText(evaluation.status)}

                    </span>

                    {evaluation.error ? (

                      <small className="error-text">

                        {getText(evaluation.error)}

                      </small>

                    ) : null}

                  </div>

                </div>

              ))

            )}

          </section>

        </div>





    <section id="playground" className="content-panel">

      <div className="section-heading">

        <div>

          <h2>Inference Playground</h2>

          <p>Generate responses using a saved LoRA adapter</p>

        </div>

        <span className="count-label">LOCAL INFERENCE</span>

      </div>



      <form onSubmit={handleGenerate} className="job-form">

        <label>

          Saved adapter

          <select

            required

            value={inferenceArtifactId}

            onChange={(event) => setInferenceArtifactId(event.target.value)}

          >

            <option value="">Select an adapter</option>

            {artifacts.map((artifact, index) => (

              <option

                key={getText(artifact.id, String(index))}

                value={getText(artifact.id, "")}

              >

                {getText(artifact.training_job_id)} —{" "}

                {getText(artifact.type, "LoRA adapter")}

              </option>

            ))}

          </select>

        </label>



        <label>

          Prompt

          <textarea

            required

            rows={5}

            maxLength={12000}

            value={inferencePrompt}

            onChange={(event) => setInferencePrompt(event.target.value)}

            placeholder="Explain how a Python function works..."

          />

        </label>



        <div className="job-form-row">

          <label>

            Max new tokens

            <input

              type="number"

              min={1}

              max={512}

              value={maxNewTokens}

              onChange={(event) =>

                setMaxNewTokens(Number(event.target.value))

              }

              required

            />

          </label>



          <label>

            Temperature (0 = deterministic)

            <input

              type="number"

              min={0}

              max={2}

              step={0.1}

              value={temperature}

              onChange={(event) =>

                setTemperature(Number(event.target.value))

              }

              required

            />

          </label>

        </div>



        {inferenceError && (

          <div className="error-banner" role="alert">

            <strong>Generation failed</strong>

            <span>{inferenceError}</span>

          </div>

        )}



        <button

          type="submit"

          className="refresh-button"

          disabled={

            generating ||

            !inferenceArtifactId ||

            !inferencePrompt.trim()

          }

        >

          {generating ? "Generating..." : "Generate response"}

        </button>

      </form>



      {generatedResponse && (

        <div className="success-banner" role="status">

          <strong>Model response</strong>

          <p style={{ whiteSpace: "pre-wrap" }}>{generatedResponse}</p>

        </div>

      )}



      {artifacts.length === 0 && (

        <p className="muted">

          Register a saved adapter before generating responses.

        </p>

      )}





<div className="comparison-panel">

  <div className="section-heading">

    <div>

      <h3>Base vs. Fine-Tuned Comparison</h3>

      <p>Run the same prompt against both model versions.</p>

    </div>

    <span className="count-label">A/B COMPARISON</span>

  </div>



  <button

    type="button"

    className="refresh-button"

    onClick={() => {

      void handleCompare({

        preventDefault: () => {},

      } as FormEvent<HTMLFormElement>);

    }}

    disabled={

      comparing ||

      generating ||

      !inferenceArtifactId ||

      !inferencePrompt.trim()

    }

  >

    {comparing ? "Comparing models..." : "Compare models"}

  </button>



  {comparisonError && (

    <div className="error-banner" role="alert">

      <strong>Comparison failed</strong>

      <span>{comparisonError}</span>

    </div>

  )}



  {comparisonResult && (

    <>

      <div className="comparison-grid">

        {(

          [

            ["base_model", "Base model"],

            ["fine_tuned_model", "Fine-tuned model"],

          ] as const

        ).map(([key, title]) => {

          const raw = comparisonResult[key];

          const result =

            typeof raw === "object" && raw !== null

              ? (raw as RecordItem)

              : {};



          const output = getText(result.response, "");



          return (

            <article className="comparison-card" key={key}>

              <div className="comparison-card-header">

                <h4>{title}</h4>

                <button

                  type="button"

                  className="refresh-button"

                  onClick={() => {

                    void navigator.clipboard.writeText(output);

                  }}

                  disabled={!output}

                >

                  Copy response

                </button>

              </div>



              <p className="muted">

                {getText(result.name, "Model")}

              </p>



              <pre className="comparison-output">

                {output || "The model returned an empty response."}

              </pre>



              <div className="comparison-stats">

                <span>

                  Time: {getText(result.generation_time_seconds, "—")} s

                </span>

                <span>

                  Tokens: {getText(result.generated_tokens, "—")}

                </span>

                <span>

                  Prompt tokens: {getText(result.prompt_tokens, "—")}

                </span>

              </div>

            </article>

          );

        })}

      </div>



      <p className="muted">

        Both models received the same prompt and token limit.

        Lower generation time does not necessarily mean better quality.

      </p>

    </>

  )}

</div>



    </section>





        <section id="inference-history" className="content-panel">
          <div className="section-heading">
            <div>
              <h2>Inference History</h2>
              <p>Previously generated responses saved in your local database</p>
            </div>
            <span className="count-label">{inferenceHistory.length} RECORDS</span>
          </div>

          {inferenceHistory.length === 0 ? (
            <p className="muted">No saved generations yet. Generate a response in the Inference Playground.</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead><tr><th>Prompt</th><th>Response</th><th>Tokens</th><th>Time</th><th>Created</th><th>Action</th></tr></thead>
                <tbody>
                  {inferenceHistory.map((item, index) => (
                    <tr key={getText(item.id, String(index))}>
                      <td>{getText(item.prompt, "").slice(0, 100)}{getText(item.prompt, "").length > 100 ? "…" : ""}</td>
                      <td>{getText(item.response, "").slice(0, 100)}{getText(item.response, "").length > 100 ? "…" : ""}</td>
                      <td>{getText(item.generated_tokens, "—")}</td>
                      <td>{getText(item.generation_time_seconds, "—")} s</td>
                      <td>{getText(item.created_at, "—")}</td>
                      <td><button type="button" className="refresh-button" onClick={async () => {
                        const id = getText(item.id, "");
                        if (!id) return;
                        const response = await fetch(`${API}/api/inference-history/${encodeURIComponent(id)}`, { method: "DELETE" });
                        if (response.ok) await refresh();
                        else setError(`Could not delete history record (HTTP ${response.status}).`);
                      }}>Delete</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section id="experiments" className="content-panel">
          <div className="section-heading">
            <div>
              <h2>Saved Experiments</h2>
              <p>Persistent base-model versus fine-tuned-model comparisons</p>
            </div>
            <span className="count-label">{experiments.length} EXPERIMENTS</span>
          </div>

          {experiments.length === 0 ? (
            <p className="muted">No comparisons saved yet. Run Compare Models in the Inference Playground.</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead><tr><th>Prompt</th><th>Base model time</th><th>Fine-tuned time</th><th>Generated tokens</th><th>Created</th><th>Action</th></tr></thead>
                <tbody>
                  {experiments.map((item, index) => (
                    <tr key={getText(item.id, String(index))}>
                      <td>{getText(item.prompt, "").slice(0, 100)}{getText(item.prompt, "").length > 100 ? "…" : ""}</td>
                      <td>{getText(item.base_generation_time_seconds, "—")} s</td>
                      <td>{getText(item.fine_tuned_generation_time_seconds, "—")} s</td>
                      <td>{getText(item.base_generated_tokens, "—")} / {getText(item.fine_tuned_generated_tokens, "—")}</td>
                      <td>{getText(item.created_at, "—")}</td>
                      <td><button type="button" className="refresh-button" onClick={async () => {
                        const id = getText(item.id, "");
                        if (!id) return;
                        const response = await fetch(`${API}/api/experiments/${encodeURIComponent(id)}`, { method: "DELETE" });
                        if (response.ok) await refresh();
                        else setError(`Could not delete experiment (HTTP ${response.status}).`);
                      }}>Delete</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section id="experiment-lab" className="content-panel" style={{ scrollMarginTop: 24 }}>
          <div className="section-heading">
            <div>
              <p className="eyebrow accent">RESEARCH NOTES / SAVED RUNS</p>
              <h2>Experiment Lab</h2>
              <p>A readable record of the model and GPU experiments completed for ForgeML.</p>
            </div>
            <span className="count-label">EXPERIMENT ARCHIVE</span>
          </div>

          <div style={{ border: "1px solid var(--border, #273244)", borderRadius: 16, padding: 20, marginBottom: 18, background: "linear-gradient(135deg, rgba(65,105,225,.10), rgba(20,184,166,.06))" }}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 18, alignItems: "center", justifyContent: "space-between" }}>
              <div style={{ flex: "1 1 320px" }}>
                <p className="eyebrow">COMPUTE ENVIRONMENT</p>
                <h3 style={{ margin: "6px 0 10px" }}>AMD Instinct MI300X · 192 GB</h3>
                <p style={{ margin: 0, lineHeight: 1.7 }}>We provisioned an AMD Developer Cloud GPU droplet named 10.1-gpu-mi300x1-192gb-devcloud-atl1. The dashboard screenshot shows it as Active with ROCm 10.1.0 on Ubuntu 24.04 and a displayed estimated rate of $1.990/hour. The saved runs below document recorded work; the screenshot verifies the cloud resource status at capture time, not that every listed experiment ran on this droplet.</p>
              </div>
              <figure style={{ flex: "1 1 560px", minWidth: 0, margin: 0, border: "1px solid rgba(120,150,190,.35)", borderRadius: 14, padding: 8, background: "rgba(8,15,28,.35)" }}>
                <a href="/amd-gpu-droplet.png" target="_blank" rel="noreferrer" aria-label="Open full AMD Developer Cloud GPU droplet screenshot" style={{ display: "block" }}>
                  <img src="/amd-gpu-droplet.png" alt="AMD Developer Cloud dashboard showing an active MI300X GPU droplet, ROCm 10.1.0 on Ubuntu 24.04, and the displayed estimated cost of $1.990 per hour" style={{ display: "block", width: "100%", height: "auto", borderRadius: 9 }} />
                </a>
                <figcaption style={{ fontSize: 12, lineHeight: 1.5, padding: "9px 5px 3px", opacity: .8 }}>Actual AMD Developer Cloud dashboard screenshot captured on October 14, 2026. The displayed rate is a snapshot and may change.</figcaption>
              </figure>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 16 }}>
              <span className="count-label">ROCm / PyTorch workflow</span><span className="count-label">LoRA fine-tuning</span><span className="count-label">Inference benchmarking</span><span className="count-label">Saved artifacts</span>
            </div>
          </div>

          <div className="section-heading"><div><h3>Recorded experiments</h3><p>Expand each card for the setup, result, interpretation, and limitations.</p></div></div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 280px), 1fr))", gap: 14 }}>
            <article style={{ border: "1px solid var(--border, #273244)", borderRadius: 14, padding: 18 }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "start" }}><span className="count-label">FINE-TUNING</span><span className="status completed">Completed</span></div>
              <h3 style={{ marginBottom: 8 }}>Qwen2.5-0.5B-Instruct · LoRA sprint</h3>
              <p style={{ lineHeight: 1.65 }}>A short fine-tuning run to validate the training path, save an adapter, and capture trainer telemetry.</p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 9, margin: "14px 0" }}>
                <div className="metric-card"><span className="metric-foot">Training steps</span><strong className="metric-value">10</strong></div><div className="metric-card"><span className="metric-foot">Epochs</span><strong className="metric-value">2</strong></div><div className="metric-card"><span className="metric-foot">Final train loss</span><strong className="metric-value">3.435</strong></div><div className="metric-card"><span className="metric-foot">Peak GPU memory</span><strong className="metric-value">~2.0 GiB</strong></div>
              </div>
              <details><summary style={{ cursor: "pointer", fontWeight: 600, padding: "8px 0" }}>Setup and what this run tells us</summary><div style={{ lineHeight: 1.7, paddingTop: 8 }}>
                <p><strong>Base model:</strong> Qwen/Qwen2.5-0.5B-Instruct.</p><p><strong>Configuration:</strong> 10 examples, 2 epochs, batch size 2, 10 steps; LoRA rank 8, alpha 16, dropout 0.05, targeting <code>q_proj</code> and <code>v_proj</code>.</p><p><strong>Trainable parameters:</strong> 540,672 of 494,573,440 (about 0.109%). The adapter was saved under <code>/workspace/forgeml/artifacts/experiments/sprint_2026-10-14/qwen_lora_sprint/adapter</code>.</p><p><strong>Interpretation:</strong> This is evidence that a small LoRA training run completed and produced a saved adapter with metrics. It is not, by itself, proof that model quality improved on unseen tasks.</p><p><strong>Limitation:</strong> The dataset and run were small; final training loss is not a standalone quality score.</p>
              </div></details>
            </article>

            <article style={{ border: "1px solid var(--border, #273244)", borderRadius: 14, padding: 18 }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "start" }}><span className="count-label">INFERENCE BENCHMARK</span><span className="status completed">Recorded</span></div>
              <h3 style={{ marginBottom: 8 }}>Qwen2.5-3B-Instruct · Generation speed</h3>
              <p style={{ lineHeight: 1.65 }}>A generation benchmark recording model startup, token throughput, and memory use on the available GPU environment.</p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 9, margin: "14px 0" }}><div className="metric-card"><span className="metric-foot">Generation</span><strong className="metric-value">58.5 tok/s</strong></div><div className="metric-card"><span className="metric-foot">Generated tokens</span><strong className="metric-value">64</strong></div><div className="metric-card"><span className="metric-foot">Generation time</span><strong className="metric-value">1.094 s</strong></div><div className="metric-card"><span className="metric-foot">Peak GPU memory</span><strong className="metric-value">5.94 GiB</strong></div></div>
              <details><summary style={{ cursor: "pointer", fontWeight: 600, padding: "8px 0" }}>Benchmark notes</summary><div style={{ lineHeight: 1.7, paddingTop: 8 }}><p><strong>Model:</strong> Qwen/Qwen2.5-3B-Instruct. The recorded run reported bfloat16, a 14.479-second model load, and a 0.702-second tokenizer load.</p><p><strong>Interpretation:</strong> A point-in-time baseline for generation throughput and memory footprint.</p><p><strong>Limitation:</strong> This is an initial measurement, not a statistically controlled comparison. Prompt length, warm-up, runtime settings, and repeated trials can change the result.</p></div></details>
            </article>

            <article style={{ border: "1px solid var(--border, #273244)", borderRadius: 14, padding: 18 }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "start" }}><span className="count-label">INFERENCE BENCHMARK</span><span className="status completed">Recorded</span></div>
              <h3 style={{ marginBottom: 8 }}>Qwen2.5-1.5B · Lightweight inference</h3>
              <p style={{ lineHeight: 1.65 }}>A smaller-model run for noting the practical memory and throughput trade-off of a lightweight model.</p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 9, margin: "14px 0" }}><div className="metric-card"><span className="metric-foot">Recorded throughput</span><strong className="metric-value">65.11 tok/s</strong></div><div className="metric-card"><span className="metric-foot">Peak GPU memory</span><strong className="metric-value">~2.96 GiB</strong></div></div>
              <details><summary style={{ cursor: "pointer", fontWeight: 600, padding: "8px 0" }}>Benchmark notes</summary><div style={{ lineHeight: 1.7, paddingTop: 8 }}><p><strong>Model family:</strong> Qwen2.5-1.5B. Saved notes also recorded an approximately 8.50-second cold model load in one run.</p><p><strong>Interpretation:</strong> This run recorded higher throughput and lower peak memory than the saved 3B run, a plausible size/performance trade-off.</p><p><strong>Limitation:</strong> These are separate initial runs, not a fully controlled benchmark. Do not treat the difference as a definitive ranking.</p></div></details>
            </article>
          </div>

          <div style={{ marginTop: 18, borderLeft: "3px solid #60a5fa", padding: "12px 16px", background: "rgba(96,165,250,.07)", borderRadius: "0 10px 10px 0" }}><strong>How to read this archive</strong><p style={{ margin: "6px 0 0", lineHeight: 1.7 }}>These cards summarize saved run notes, not live GPU telemetry. Metrics describe the specific recorded runs and may vary on rerun. For reproducibility, keep original logs, model identifiers, dataset, training configuration, and adapter files together. Use the live Training Jobs, Artifacts, and Inference History sections above for records returned by the running backend.</p></div>
        </section>

        <footer className="footer">

          <span>ForgeML · Local development</span>

          <span>

            {error

              ? "API disconnected"

              : `Last synced: ${lastUpdated || "connecting…"}`}

          </span>

        </footer>

      </section>

    </main>

  );

}
