
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
};

const endpoints = [
  { key: "models", label: "Models", path: "/api/models/" },
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

    setGeneratedResponse(result.response ?? "");
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

    setComparisonResult(result as RecordItem);
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
    const timer = window.setInterval(() => void refresh(), 5000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const models = data.models ?? [];
  const datasets = data.datasets ?? [];
  const jobs = data.jobs ?? [];
  const artifacts = data.artifacts ?? [];
  const evaluations = data.evaluations ?? [];

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
