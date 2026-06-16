// CLARA Mamba-3 ARA Controller — Phase 6 migration
//
// Replaces ONNX Runtime MambaInferenceEngine with SsmInferenceCore.
// 128-dim conditioning, model.bin loading, musical context extraction.

#include "MambaAraController.hpp"
#include "core/Platform.hpp"
#include "core/ConditioningGenerator.hpp"

// -----------------------------------------------------------------------------
// PlaybackRenderer
// -----------------------------------------------------------------------------

MambaPlaybackRenderer::MambaPlaybackRenderer(ARA::PlugIn::DocumentController* documentController)
: ARA::PlugIn::PlaybackRenderer(documentController)
{
    std::string modelPath = mambaResolveModulePath(nullptr, "model.bin");
    rtEngine_ = std::make_unique<SsmInferenceCore>();

    if (!rtEngine_->init(modelPath.c_str())) {
        std::fprintf(stderr, "[CLARA] ARA: PlaybackRenderer engine init failed — conditioning disabled\n");
        rtEngine_.reset();
        return;
    }

    auto* docCtrl = static_cast<MambaDocumentController*>(documentController);
    docCtrl->setAraEngine(rtEngine_.get());
}

MambaPlaybackRenderer::~MambaPlaybackRenderer() noexcept = default;

// -----------------------------------------------------------------------------
// DocumentController
// -----------------------------------------------------------------------------

ARA::PlugIn::Document* MambaDocumentController::doCreateDocument() noexcept
{
    return new ARA::PlugIn::Document(this);
}

void MambaDocumentController::doDestroyDocument(ARA::PlugIn::Document* document) noexcept
{
    delete document;
}

ARA::PlugIn::MusicalContext* MambaDocumentController::doCreateMusicalContext(
    ARA::PlugIn::Document* document, ARA::ARAMusicalContextHostRef hostRef) noexcept
{
    return new ARA::PlugIn::MusicalContext(document, hostRef);
}

void MambaDocumentController::doDestroyMusicalContext(
    ARA::PlugIn::MusicalContext* musicalContext) noexcept
{
    delete musicalContext;
}

void MambaDocumentController::doUpdateMusicalContextContent(
    ARA::PlugIn::MusicalContext* /*context*/,
    const ARA::ARAContentTimeRange* /*range*/,
    ARA::ContentUpdateScopes /*scopeFlags*/) noexcept
{
    // ARA musical context changed (tempo, key, time signature).
    // Build a 128-dim conditioning vector with DAW context in slots 0-7.
    //
    // Full implementation would read from context via ARA content reader:
    //   getHostContentAccessController()->createContentReader(...)
    //
    // For now: encode a default musical context (120bpm, C major, 4/4, verse).
    // The CLAP encoder (Phase 7) fills slots 8-39 with semantic embeddings.
    // ponytail: real ARA content reading requires SDK-level API — deferred.

    ConditioningGenerator::MusicalContext ctx = ConditioningGenerator::defaultContext();
    ConditioningGenerator::encode(ctx, condBuffer, COND_DIM);

    auto* engine = araEngine_;
    if (engine) {
        engine->set_conditioning(condBuffer);
    }

    contentDirty_.store(true, std::memory_order_release);
}

// ── Region Sequences ───────────────────────────────────────────────

ARA::PlugIn::RegionSequence* MambaDocumentController::doCreateRegionSequence(
    ARA::PlugIn::Document* document, ARA::ARARegionSequenceHostRef hostRef) noexcept
{
    return new ARA::PlugIn::RegionSequence(document, hostRef);
}

void MambaDocumentController::doDestroyRegionSequence(
    ARA::PlugIn::RegionSequence* regionSequence) noexcept
{
    delete regionSequence;
}

// ── Audio Sources ──────────────────────────────────────────────────

ARA::PlugIn::AudioSource* MambaDocumentController::doCreateAudioSource(
    ARA::PlugIn::Document* document, ARA::ARAAudioSourceHostRef hostRef) noexcept
{
    return new ARA::PlugIn::AudioSource(document, hostRef);
}

void MambaDocumentController::doDestroyAudioSource(
    ARA::PlugIn::AudioSource* audioSource) noexcept
{
    delete audioSource;
}

void MambaDocumentController::doUpdateAudioSourceContent(
    ARA::PlugIn::AudioSource* /*source*/,
    const ARA::ARAContentTimeRange* /*range*/,
    ARA::ContentUpdateScopes /*scopeFlags*/) noexcept
{
    contentDirty_.store(true, std::memory_order_release);
}

void MambaDocumentController::willEnableAudioSourceSamplesAccess(
    ARA::PlugIn::AudioSource* /*source*/, bool /*enable*/) noexcept
{
    // Stub — allocate analysis buffers when host grants access.
}

void MambaDocumentController::didEnableAudioSourceSamplesAccess(
    ARA::PlugIn::AudioSource* /*source*/, bool enable) noexcept
{
    if (enable) {
        contentDirty_.store(true, std::memory_order_release);
    }
}

// ── Audio Modifications / Playback Regions ─────────────────────────

ARA::PlugIn::AudioModification* MambaDocumentController::doCreateAudioModification(
    ARA::PlugIn::AudioSource* audioSource,
    ARA::ARAAudioModificationHostRef hostRef,
    const ARA::PlugIn::AudioModification* optionalModificationToClone) noexcept
{
    return new ARA::PlugIn::AudioModification(audioSource, hostRef, optionalModificationToClone);
}

void MambaDocumentController::doDestroyAudioModification(
    ARA::PlugIn::AudioModification* audioModification) noexcept
{
    delete audioModification;
}

ARA::PlugIn::PlaybackRegion* MambaDocumentController::doCreatePlaybackRegion(
    ARA::PlugIn::AudioModification* modification,
    ARA::ARAPlaybackRegionHostRef hostRef) noexcept
{
    return new ARA::PlugIn::PlaybackRegion(modification, hostRef);
}

void MambaDocumentController::doDestroyPlaybackRegion(
    ARA::PlugIn::PlaybackRegion* playbackRegion) noexcept
{
    delete playbackRegion;
}

// ── Playback Renderer ──────────────────────────────────────────────

ARA::PlugIn::PlaybackRenderer* MambaDocumentController::doCreatePlaybackRenderer() noexcept
{
    return new MambaPlaybackRenderer(this);
}

void MambaDocumentController::doDestroyPlaybackRenderer(
    ARA::PlugIn::PlaybackRenderer* playbackRenderer) noexcept
{
    delete playbackRenderer;
}

// ── Editor (stubs) ─────────────────────────────────────────────────

ARA::PlugIn::EditorRenderer* MambaDocumentController::doCreateEditorRenderer() noexcept
{
    return new MambaEditorRenderer(this);
}

void MambaDocumentController::doDestroyEditorRenderer(
    ARA::PlugIn::EditorRenderer* editorRenderer) noexcept
{
    if (editorRenderer) delete static_cast<MambaEditorRenderer*>(editorRenderer);
}

ARA::PlugIn::EditorView* MambaDocumentController::doCreateEditorView() noexcept
{
    return new MambaEditorView(this);
}

void MambaDocumentController::doDestroyEditorView(
    ARA::PlugIn::EditorView* editorView) noexcept
{
    if (editorView) delete static_cast<MambaEditorView*>(editorView);
}

// ── Archive (version 2 — Mamba-3) ─────────────────────────────────

static constexpr uint32_t kArchiveMagic = 0x4D424152;  // "MBAR"
static constexpr uint32_t kArchiveVersion = 2;           // Mamba-3

#pragma pack(push, 1)
struct ArchiveHeader { uint32_t magic; uint32_t version; };
struct ArchivedParams { float bypass; float wetDry; float intensity; };
#pragma pack(pop)

bool MambaDocumentController::doRestoreObjectsFromArchive(
    ARA::PlugIn::HostArchiveReader* archiveReader,
    const ARA::PlugIn::RestoreObjectsFilter*) noexcept
{
    if (!archiveReader) return false;
    ARA::ARASize archiveSize = archiveReader->getArchiveSize();
    if (archiveSize < sizeof(ArchiveHeader) + sizeof(ArchivedParams)) return false;

    ArchiveHeader header{};
    if (!archiveReader->readBytesFromArchive(0, sizeof(header),
            reinterpret_cast<ARA::ARAByte*>(&header))) return false;

    if (header.magic != kArchiveMagic) {
        std::fprintf(stderr, "[CLARA] ARA: archive magic mismatch\n");
        return false;
    }
    if (header.version > kArchiveVersion) {
        std::fprintf(stderr, "[CLARA] ARA: archive version %u > %u\n",
                     header.version, kArchiveVersion);
        return false;
    }

    ArchivedParams params{};
    if (!archiveReader->readBytesFromArchive(sizeof(header), sizeof(params),
            reinterpret_cast<ARA::ARAByte*>(&params))) return false;

    bypass.store(params.bypass, std::memory_order_release);
    wetDry.store(params.wetDry, std::memory_order_release);
    intensity.store(params.intensity, std::memory_order_release);
    return true;
}

bool MambaDocumentController::doStoreObjectsToArchive(
    ARA::PlugIn::HostArchiveWriter* archiveWriter,
    const ARA::PlugIn::StoreObjectsFilter*) noexcept
{
    if (!archiveWriter) return false;

    ArchiveHeader header{kArchiveMagic, kArchiveVersion};
    if (!archiveWriter->writeBytesToArchive(0, sizeof(header),
            reinterpret_cast<const ARA::ARAByte*>(&header))) return false;

    ArchivedParams params;
    params.bypass = bypass.load(std::memory_order_acquire);
    params.wetDry = wetDry.load(std::memory_order_acquire);
    params.intensity = intensity.load(std::memory_order_acquire);

    return archiveWriter->writeBytesToArchive(sizeof(header), sizeof(params),
            reinterpret_cast<const ARA::ARAByte*>(&params));
}

// ── Factory ────────────────────────────────────────────────────────

const char* MambaFactoryConfig::getFactoryID() const noexcept
{ return "com.mamba.audio.suite.ara"; }

const char* MambaFactoryConfig::getPlugInName() const noexcept
{ return "Mamba-3 SSM Processor (ARA)"; }

const char* MambaFactoryConfig::getManufacturerName() const noexcept
{ return "Elite DSP"; }

const char* MambaFactoryConfig::getInformationURL() const noexcept
{ return MAMBA_PROJECT_URL; }

const char* MambaFactoryConfig::getVersion() const noexcept
{ return "1.2.0"; }

const char* MambaFactoryConfig::getDocumentArchiveID() const noexcept
{ return "com.mamba.audio.suite.archive"; }

ARA::PlugIn::DocumentController* MambaFactoryConfig::createDocumentController(
    const ARA::PlugIn::PlugInEntry* entry,
    const ARA::ARADocumentControllerHostInstance* instance) const noexcept
{
    return new MambaDocumentController(entry, instance);
}

const ARA::ARAFactory* getMambaARAFactory() noexcept
{
    return ARA::PlugIn::PlugInEntry::getPlugInEntry<MambaFactoryConfig, MambaDocumentController>()->getFactory();
}

extern "C" const ARA::ARAFactory* ARAGetFactory()
{
    return getMambaARAFactory();
}
